from PIL import ImageDraw
import argparse
import datetime
import gradio as gr
import hashlib
import json
import numpy as np
import os
import re
import requests
import time
from robopoint.conversation import (default_conversation, conv_templates, SeparatorStyle)
from robopoint.constants import LOGDIR
from robopoint.utils import (
    build_logger, server_error_msg, violates_moderation, moderation_msg
)


logger = build_logger("gradio_web_server", "gradio_web_server.log")

headers = {"User-Agent": "RoboPoint Client"}

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown(
        choices=models,
        value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def add_text(state, text, image, image_process_mode, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (
                state, state.to_gradio_chatbot(), moderation_msg, None
            ) + (no_change_btn,) * 5

    text += " Your answer should be formatted as a list of tuples, " \
            "i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the " \
            "x and y coordinates of a point satisfying the conditions above." \
            " The coordinates should be between 0 and 1, indicating the " \
            "normalized pixel locations of the points in the image."
    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            text = '<image>\n' + text
        text = (text, image, image_process_mode)
        state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def find_vectors(text):
    # text = text.split('<ans>')[1].split('</ans>')[0]

    # This pattern matches lists of integers or floating-point numbers, including negatives
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"

    matches = re.findall(pattern, text)

    # Convert string matches to lists of floats or ints
    vectors = []
    for match in matches:
        # Split each match by comma and convert to appropriate numeric type
        vector = [float(num) if '.' in num else int(num) for num in match.split(',')]
        vectors.append(vector)

    return vectors


def visualize_2d(img, points, bounding_boxes, scale, cross_size=9, cross_width=4):
    # msg_data is a tuple: (PIL Image, image_mode)
    # image_mode has something to do with how the image is cropped when feeding into model

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Create a drawing context
    draw = ImageDraw.Draw(img)
    size = int(cross_size * scale)
    width = int(cross_width * scale)
    logger.info(f"marker scale {scale} size {size} width {width}")

    # Draw each point as a red X
    for x, y in points:
        # Draw a cross ('X') at the point location
        draw.line((
            x - size, y - size, x + size, y + size
        ), fill='red', width=width)
        draw.line((
            x - size, y + size, x + size, y - size
        ), fill='red', width=width)

    # Draw each bounding box as a red rectangle
    for x1, y1, x2, y2 in bounding_boxes:
        # Draw the rectangle outline
        draw.rectangle([x1, y1, x2, y2], outline='red', width=width)

    img = img.convert('RGB')
    return img


def visualize_response(img_data, transform, query, response, ref_w=640, ref_h=480):
    print("visualizing response")
    assert 'user' in query[0].lower() or 'human' in query[0].lower() \
        and 'assistant' in response[0].lower(), f"{query[0]} {response[0]} " \
        "Expected query and response to be from user and assistant."

    vectors = find_vectors(response[1])
    vectors_2d = [vec for vec in vectors if len(vec) == 2]   # 2d points
    vectors_bbox = [vec for vec in vectors if len(vec) == 4] # bounding boxes
    print(
        f"Found:\n  - {len(vectors_2d)} 2D vectors.\n"
        f"  - {len(vectors_bbox)} 2D bounding boxes."
    )
    vectors_other = [vec for vec in vectors if len(vec) not in [2, 4]]
    if len(vectors_other) > 0:
        print("Found vectors that are neither 2d points nor bounding boxes:")
        print(vectors_other)
        print("   ^ these will not be visualized")
    if len(vectors_2d) + len(vectors_bbox) == 0:
        return response

    # Find the image to annotate
    _, img, process_mode = img_data

    # Resize points
    new_vectors = []
    for x, y in vectors_2d:
        if isinstance(x, float) and x <= 1:
            x = x * ref_w
            y = y * ref_h
        print('scaled', (x, y))
        x, y, _ = (transform @ np.array([[x], [y], [1]])).ravel()
        print('transformed', (x, y))
        new_vectors.append((x, y))
    vectors_2d = new_vectors

    # Resize bounding boxes
    new_bbox = []
    for x1, y1, x2, y2 in vectors_bbox:
        # Calculate the top left and bottom right coordinates of the box
        if isinstance(x1, float) and x1 <= 1:
            x1 = x1 * ref_w
            y1 = y1 * ref_h
            x2 = x2 * ref_w
            y2 = y2 * ref_h
        x1, y1, _ = (transform @ np.array([[x1], [y1], [1]])).ravel()
        x2, y2, _ = (transform @ np.array([[x2], [y2], [1]])).ravel()
        new_bbox.append((x1, y1, x2, y2))

    # Plot the annotations on the image
    anno_img = visualize_2d(img, vectors_2d, vectors_bbox, transform[0][0])
    # Construct a multimodal image response to display in gradio chat and replace the original text message.
    new_response = [response[0], (response[1], anno_img, process_mode)]
    return new_response


def http_bot(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if 'vicuna' in model_name.lower():
            template_name = "vicuna_v1"
        elif "llama" in model_name.lower():
            template_name = "llava_llama_2"
        elif "mistral" in model_name.lower():
            template_name = "mistral_instruct"
        elif "mpt" in model_name.lower():
            template_name = "mpt"
        else:
            template_name = "llava_v1"

        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model": model_name})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    prompt = state.get_prompt()
    pil_images, images, transforms = state.get_images()
    image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in pil_images]
    for image, hash in zip(pil_images, image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg")
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images": f'List of {len(pil_images)} images: {image_hash}',
    }
    logger.info(f"==== request ====\n{pload}")
    pload['images'] = images

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(worker_addr + "/worker_generate_stream",
            headers=headers, json=pload, stream=True, timeout=10)
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt):].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
                    return
                # time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (disable_btn, disable_btn, disable_btn, enable_btn, enable_btn)
        return

    # Find query image
    img_data = None
    for j in range(len(state.messages) - 1, -1, -1):
        role, content = state.messages[j]
        if 'user' in role.lower() or 'human' in role.lower():
            if isinstance(content, tuple) or isinstance(content, list):
                img_data = content
                break
    if img_data is not None:
        state.messages[-1] = visualize_response(
            img_data, transforms[-1], state.messages[-2], state.messages[-1]
        )
    if isinstance(state.messages[-1][1], tuple):
        image = state.messages[-1][1][1]
        hash = hashlib.md5(image.tobytes()).hexdigest()
        t = datetime.datetime.now()
        filename = os.path.join(
            LOGDIR, "response_images", f"{t.year}-{t.month:02d}-{t.day:02d}",
            f"{hash}-{t.hour}-{t.minute}-{t.second}.jpg"
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        image.save(filename)
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")

title_markdown = ("""
# RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics
[[Project Page](https://robo-point.github.io)] | [[Paper](https://arxiv.org/abs/2406.10721)]
""")

tos_markdown = ("""
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.
Please click the "Flag" button if you get any inappropriate answer! We will collect those to keep improving our moderator.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
""")


learn_more_markdown = ("""
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
""")

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""

def build_demo(embed_mode, cur_dir=None, concurrency_count=10):
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False)

                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Pad"], value="Pad", label="Preprocessing for images"
                )

                if cur_dir is None:
                    cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/sink.jpg", "Find a few spots within the vacant area on the rightmost white plate."],
                    [f"{cur_dir}/examples/stair.jpg", "Identify several places in the unoccupied space on the stair in the middle."],
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=True) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="RoboPoint Chatbot",
                    height=650,
                    layout="panel"
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        if not embed_mode:
            gr.Markdown(tos_markdown)
            # gr.Markdown(learn_more_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn]
        )

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list
        ).then(
            http_bot,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
            concurrency_limit=concurrency_count
        )

        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                js=get_window_url_params
            )
        elif args.model_list_mode == "reload":
            demo.load(
                load_demo_refresh_model_list,
                None,
                [state, model_selector],
                queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=16)
    parser.add_argument("--model-list-mode", type=str, default="once",
        choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo(args.embed, concurrency_count=args.concurrency_count)
    demo.queue(
        api_open=False
    ).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )

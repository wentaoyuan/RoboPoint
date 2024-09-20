from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import argparse
import base64
import json
import openai
# Insert your OpenAI API key here!!!
openai.api_key = ''
from time import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


system = \
    "You are an agent who reasons about free space and spatial relations. " \
    "Based on the given image and prompt, your task is to output a tuple " \
    "(min_x, min_y, max_x, max_y) that represents the bounding box of the " \
    "object or region in the image that best matches the description in the " \
    "prompt. min_x, min_y, max_x, max_y should be integers indicating pixel " \
    "locations. Do not include any additional text in your answer other than " \
    "the tuple (min_x, min_y, max_x, max_y)."
prompt = "Here is an image whose dimensions are labeled on the side."


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def completion_with_backoff(**kwargs):
    st = time()
    response = openai.chat.completions.create(**kwargs)
    print(f"GPT responded in {time() - st:.2f}s")
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--data_dir", default="/home/wentaoy/datasets/where2place")
    parser.add_argument("--question_file", default="bbox_questions.jsonl")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_questions", type=int)
    parser.add_argument("--answer_file")
    args = parser.parse_args()

    with open(f"{args.data_dir}/{args.question_file}", 'r') as file:
        questions = [json.loads(line) for line in file]
    for idx, question in enumerate(questions[:args.num_questions]):
        q_txt = question['text'].split(' Your answer')[0]
        if '<image>' in q_txt:
            q_txt = q_txt.split('\n')[1]
        fig, ax = plt.subplots()
        img = Image.open(f"{args.data_dir}/images/{question['image']}")
        plt.imshow(img)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        plt.tight_layout()
        plt.show(block=False)
        buffer = BytesIO()
        plt.savefig(buffer, format="JPEG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        messages = [{
            "role": "system",
            "content": [{"type": "text", "text": system}]
        }, {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt} {q_txt}"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }}
            ]
        }]
        model = 'gpt-4-vision-preview' if args.model == 'gpt-4v' else args.model
        response = completion_with_backoff(
            model=model, messages=messages, temperature=args.temperature
        )
        answer = response.choices[0].message.content
        print(answer)
        answer = {
            'question_id': idx,
            'prompt': q_txt,
            'text': answer,
            'answer_id': response.id,
            'model_id': model,
            'category': question['category']
        }
        with open(args.answer_file, 'a') as f:
            f.write(json.dumps(answer) + '\n')

from PIL import Image
from matplotlib import colormaps
from matplotlib import patches
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import json
import numpy as np
import os
import re


def find_vectors(text, width=640, height=480):
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    vectors, points = [], []
    for match in matches:
        vector = [
            float(num) if '.' in num else int(num) for num in match.split(',')
        ]
        if len(vector) == 2:
            x, y = vector
            if isinstance(x, float) or isinstance(y, float):
                x = int(x * width)
                y = int(y * height)
            vectors.append((x, y))
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0, x1, y1 = vector
            if isinstance(x0, float):
                x0 = int(x0 * width)
                y0 = int(y0 * height)
                x1 = int(x1 * width)
                y1 = int(y1 * height)
            vectors.append((x0, y0, x1, y1))
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = 1
            y, x = np.where(mask)
            points.extend(list(np.stack([x, y], axis=1)))
    return vectors, np.array(points)


def word_wrap(s, n):
    words = s.split()
    for i in range(n, len(words), n):
        words[i-1] += '\n'
    return ' '.join(words)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_files", nargs='+')
    parser.add_argument("--labels", nargs='+')
    parser.add_argument("--data_dir", default="/home/wentaoy/datasets/where2place")
    parser.add_argument("--question_file", default="point_questions.jsonl")
    parser.add_argument("--num_questions", type=int)
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    with open(f"{args.data_dir}/{args.question_file}", 'r') as file:
        questions = [json.loads(line) for line in file]
    if args.num_questions is None:
        args.num_questions = len(questions)
    colors = colormaps['Set1']
    answers = {}
    for fname, label in zip(args.answer_files, args.labels):
        with open(fname, 'r') as file:
            answers[label] = [json.loads(line) for line in file]
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, question in enumerate(tqdm(questions[:args.num_questions])):
        img = Image.open(f"{args.data_dir}/images/{question['image']}")
        mask = np.array(Image.open(f"{args.data_dir}/masks/{idx:02d}.jpg")) / 255.
        labels, vectors = ['Ground Truth'], {}
        for key, ans in answers.items():
            try:
                vectors[key], pts = find_vectors(ans[idx]['text'])
            except:
                print('Failed to parse answer for question', idx, 'from', key)
                pts = []
            acc = 0
            if len(pts) > 0:
                in_range = (pts[:, 0] >= 0) & (pts[:, 0] < mask.shape[1]) \
                         & (pts[:, 1] >= 0) & (pts[:, 1] < mask.shape[0])
                acc = np.concatenate([
                    mask[pts[in_range, 1], pts[in_range, 0]],
                    np.zeros(pts.shape[0] - in_range.sum())
                ]).mean()
            labels.append(f"{key}\nAccuracy: {acc:.2f}")
        fig, ax = plt.subplots(1, figsize=(10.2, 6.6))
        plt.imshow(img)
        colored_mask = np.ones_like(mask)[..., None] * colors(0)
        colored_mask[..., 3] = mask * 0.6
        plt.imshow(colored_mask)
        plt.axis('off')
        for i, vec in enumerate(vectors.values()):
            for v in vec:
                if len(v) == 2:
                    x, y = v
                    c1 = patches.Circle((x, y), 3, color=colors(i+1), fill=True)
                    ax.add_patch(c1)
                    c2 = patches.Circle((x, y), 10, color=colors(i+1), fill=False)
                    ax.add_patch(c2)
                elif len(v) == 4:
                    x0, y0, x1, y1 = v
                    rect = patches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0, linewidth=2,
                        edgecolor=colors(i+1), facecolor='none'
                    )
                    ax.add_patch(rect)
        handles = [
            patches.Patch(color=colors(i), label=label)
            for i, label in enumerate(labels)
        ]
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.005, 1))
        q_txt = question['text'].split("Your answer should")[0].strip()
        if '<image>' in q_txt:
            q_txt = q_txt.split('\n')[1]
        plt.suptitle(q_txt, y=0.995)
        plt.subplots_adjust(left=0.005, right=0.83, bottom=0.01, top=0.955)
        img_name = question['image'].split('/')[-1]
        img_path = f"{args.output_dir}/{img_name}"
        plt.savefig(img_path)
        plt.close()

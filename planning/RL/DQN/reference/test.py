import argparse
import torch
import os
from src import Tetris
from utils import imgs2gif

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", type=bool, default=True)

    parser.add_argument("--render", type=bool, default=True, help='flag - video render')
    parser.add_argument("--fps", type=int, default=300, help="frames per second")

    parser.add_argument("--width", type=int, default=8, help="The common width for all images")
    parser.add_argument("--height", type=int, default=8, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=8, help="Size of a block")
    parser.add_argument("--save_frame", type=bool, default=True)
    parser.add_argument("--convert_gif", type=bool, default=True)

    weightFile = 'tetris_E2322_L491'
    parser.add_argument("--saved_path", type=str, default="weightFiles/"+weightFile)
    parser.add_argument("--out_images", type=str, default='out_images/'+weightFile)

    args = parser.parse_args()
    return args


def test(opt):
    os.makedirs(opt.out_images) if not os.path.isdir(opt.out_images) else None

    # seed & device
    torch.manual_seed(opt.seed)
    device = 'cuda' if opt.use_cuda and torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.manual_seed_all(opt.seed)

    # model
    model = torch.load(opt.saved_path).to(device)
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size, out_images=opt.out_images)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=opt.render, save_frame=opt.save_frame)

        if done:
            score = env.score
            num_pieces = env.num_pieces
            cleared_lines = env.cleared_lines
            print(f' Score: {score}  |  Cleared Lines: {cleared_lines}  |  N Pieces: {num_pieces}\n')
            break
    if opt.convert_gif:
        print('converting to gif...')
        imgs2gif(opt.out_images, loop=1)

    return cleared_lines


if __name__ == "__main__":
    opt = get_args()
    test(opt)

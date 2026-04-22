import os


def imgs2gif(dir_path, loop=0, resize=(250, 625)):
    os.system(f'convert -resize {resize[0]}x{resize[1]} -delay 1 -loop {loop} {dir_path}/*.jpg {dir_path}.gif')
    print('Success Converting! >>', f'{dir_path}.gif')


if __name__ == '__main__':
    dir_path = '../out_images/tetris_3651_741'
    imgs2gif(dir_path, loop=1)


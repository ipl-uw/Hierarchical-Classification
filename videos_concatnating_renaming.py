from tqdm import tqdm
import argparse
import os
from natsort import natsorted

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hierarchical classification")
    parser.add_argument('-video_folders', '--video_folders', nargs='*', help='<Required> Set flag', required=True)
    parser.add_argument('-n_digits', '--n_digits', type=int, help='number of digits', required=True)

    args = parser.parse_args()


    num = 0
    for video_folder in tqdm(args.video_folders):

        frames = os.listdir(video_folder)
        frames = natsorted(frames)

        for frame in frames:

            frame_id = str(num).zfill(args.n_digits)
            os.rename(os.path.join(video_folder,frame), os.path.join(video_folder,frame_id+'.jpg'))
            num += 1

    print('total frames: %d' %num)

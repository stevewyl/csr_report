# 使用HanLP的Python接口进行分词
from pyhanlp import HanLP
from pathlib import Path
from utils import save_line, read_line


def cut_line(line_text):
        return [str(HanLP.segment(line)) for line in line_text if line != '']


if __name__ == '__main__':
    if not Path('segmented').is_dir():
        Path('segmented').mkdir()

    for i in range(2002, 2017):
        print('reading files from folder', str(i))
        input_path = Path(__file__).parent / 'cleaned' / str(i)
        output_path = Path(__file__).parent / 'segmented' / str(i)
        if not output_path.exists():
            output_path.mkdir()
        input_files = input_path.glob('*.txt')
        for file in input_files:
            segmented_text = cut_line(read_line(file))
            save_line(segmented_text, output_path.joinpath(file.name))

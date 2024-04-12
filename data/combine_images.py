import os
import json
from argparse import ArgumentParser
from PIL import Image, ImageDraw, ImageFont
import math
from tqdm import tqdm


def resize_image(image, target_size):
    """
    Resize the given image to the target size.
    
    Parameters:
        image (PIL.Image): The input image to be resized.
        target_size (tuple): A tuple representing the target size (width, height).
    
    Returns:
        PIL.Image: The resized image.
    """
    return image.resize(target_size, Image.LANCZOS).convert('RGBA')

def add_caption(image, caption_text, font_path='arialuni.ttf', font_size=15, caption_color='white', background_color='black'):
    """
    Add caption to the image.
    
    Parameters:
        image (PIL.Image): The input image to which the caption will be added.
        caption_text (str): The text of the caption.
        font_path (str): The file path of the font.
        font_size (int): The font size for the caption.
        caption_color (str): Color str for the caption.
        background_color (str): Color str for the background of the index.
    
    Returns:
        PIL.Image: The image with added caption.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    bbox = draw.textbbox((0, 0), caption_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    image_width, image_height = image.size
    x = (image_width - text_width) / 2
    y = image_height - text_height - 10
    
    # Enlarge the font for the index
    index_font = ImageFont.truetype(font_path, font_size * 4)
    index_text = caption_text
    bbox = draw.textbbox((0, 0), index_text, font=index_font)
    index_text_width = bbox[2] - bbox[0]
    index_text_height = bbox[3] - bbox[1]
    index_text_position = ((image_width - index_text_width) // 2, (image_height - index_text_height))

    # Calculate the position and size of the rectangular box
    padding = 12
    rect_left = index_text_position[0] - padding
    rect_top = index_text_position[1] - padding - 10
    rect_right = index_text_position[0] + index_text_width + padding
    rect_bottom = index_text_position[1] + index_text_height + padding
    # Draw text and rectangle
    draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], fill=background_color)
    draw.text(index_text_position, index_text, fill=caption_color, font=index_font)

    return image

def combine_images(image_paths, font_size=15, img_start_index=0, resize_to=(800, 800)):
    """
    Combine multiple images into one grid and save the result.
    
    Parameters:
        image_paths (list): List of file paths of the input images.
        resize_to (tuple): A tuple representing the target size for each image in the grid (width, height).
    """
    images = [resize_image(Image.open(path), resize_to) for path in image_paths]
    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))
    grid_width = grid_size
    grid_height = math.ceil(num_images / grid_width)
    total_width = grid_width * resize_to[0]
    total_height = grid_height * resize_to[1]
    
    new_image = Image.new('RGB', (total_width, total_height))

    x_offset = 0
    y_offset = 0
    for i, img in enumerate(images):
        caption = f"Image {img_start_index+i+1}"
        img_with_caption = add_caption(img, caption, font_size=font_size)
        new_image.paste(img_with_caption, (x_offset, y_offset))
        x_offset += resize_to[0]
        if x_offset >= total_width:
            x_offset = 0
            y_offset += resize_to[1]
    return new_image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='data/MileBench')
    parser.add_argument('--dataset_name', default='data/sample.json')
    parser.add_argument('--max_img_num', required=True, type=int, help='Max num of images model can handle')
    parser.add_argument('--font_size', default=8, type=int)
    args = parser.parse_args()
    return args

def main(args):
    
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    max_img_num = args.max_img_num

    dataset_dir = os.path.join(data_dir, dataset_name)
    img_dir = os.path.join(dataset_dir, 'images')
    output_dir = os.path.join(dataset_dir, f'combined_{max_img_num}_images')
    os.makedirs(output_dir, exist_ok=True)

    json_data = json.load(open(os.path.join(dataset_dir, f'{dataset_name}.json')))
    for item in tqdm(json_data['data']):
        # Generate combined image
        image_file_names = item['task_instance']['images_path']
        image_file_paths = [os.path.join(img_dir, name) for name in image_file_names]
        
        # number of old-imgs in each new-img
        # NOTE: if num of image_file_paths less than max_img_num, combine_num=1, just index the img
        combine_num = (len(image_file_paths)-1)//max_img_num + 1

        # start combining imgs
        img_output_paths = []
        for idx, start_idx in enumerate(range(0, len(image_file_paths), combine_num)):
            output_path = os.path.join(output_dir, f'{item["sample_id"]}-{idx}.jpg')
            img_output_paths.append(f'{item["sample_id"]}-{idx}.jpg')
            new_image = combine_images(
                image_paths=image_file_paths[start_idx:start_idx+combine_num],
                font_size=args.font_size, 
                img_start_index=start_idx, 
                resize_to=(800, 800))
            # Save the new img
            new_image.save(output_path)
        
        # Merge to json-data file
        item['task_instance'][f"combined_{max_img_num}_images"] = img_output_paths

    # Save json-data
    json.dump(
        json_data, 
        open(os.path.join(dataset_dir, f'{dataset_name}_combined_{max_img_num}.json'), 'w'),
        ensure_ascii=False, indent=4)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)

'''
python combine_images.py \
    --data_dir MileBench \
    --dataset_name ALFRED \
    --max_img_num 3 \
    --font_size 8
'''
import json

from argument_parsing import parse_command_line_args_filter


def main():
    args = parse_command_line_args_filter()

    category = args.category
    is_crowd_include = args.is_crowd_include
    min_bbox_area_ratio = args.min_bbox_area_ratio
    input_path = args.input_path
    output_path = args.output_path

    with open(input_path) as json_file:
        input_data = json.load(json_file)

    result = {
        'images': [],
        'annotations': [],
        'categories': input_data['categories']
    }

    for annotation in input_data['annotations']:
        image_id = annotation['image_id']
        image = next((x for x in input_data['images'] if x['id'] == image_id), None)
        image_in_result = next((x for x in result['images'] if x['id'] == image_id), None)
        bbox_area_ratio = annotation['area'] / (image['height'] * image['width'])

        if (
            annotation['category_id'] == category and \
            annotation['iscrowd'] == is_crowd_include and \
            bbox_area_ratio > min_bbox_area_ratio
           ):
            result['annotations'].append(annotation)
            if image_in_result is None:
                result['images'].append(image)
                
    with open(output_path, 'w') as json_file:
        json.dump(result, json_file) 


if __name__ == '__main__':
    main()
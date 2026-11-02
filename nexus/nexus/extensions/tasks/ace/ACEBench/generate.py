import argparse
import json
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

from category import ACE_DATA_CATEGORY
from model_inference.inference_map import inference_map
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o', nargs='+', help='Name of the model(s) to use')
    parser.add_argument(
        '--category', type=str, default='test_all', nargs='+', help='Category of the model you want to test'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.7, help='Temperature parameter to control randomness of model output'
    )
    parser.add_argument('--top-p', type=float, default=1, help='Top-p parameter to control diversity of model output')
    parser.add_argument('--max-tokens', type=int, default=1200, help='Maximum number of tokens to generate')
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        choices=['en', 'zh'],
        help="Language for model output, choose 'en' for English or 'zh' for Chinese",
    )
    parser.add_argument('--num-threads', type=int, default=1, help='Number of threads to use for concurrency')
    parser.add_argument(
        '--max-dialog-turns', type=int, default=40, help='Maximum number of dialog turns allowed for agent interactions'
    )
    parser.add_argument('--user-model', type=str, default='gpt-4o', help='Model used by the user role in the agent')
    return parser.parse_args()


def load_test_cases(base_path, filenames):
    cases = []

    for filename in filenames:
        file_path = os.path.join(base_path, filename)
        try:
            with open(file_path, encoding='utf-8') as file:
                cases.extend(json.loads(line) for line in file)
        except FileNotFoundError:
            print(f'Error: File not found - {file_path}')
        except json.JSONDecodeError:
            print(f'Error: Failed to parse JSON in file - {file_path}')
    return cases


def sort_json(file):
    data = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    if 'multi_turn' in file and 'agent' not in file:
        data = sorted(data, key=lambda x: tuple(map(int, x['id'].split('_')[-2:])))
    else:
        data = sorted(data, key=lambda x: int(x['id'].split('_')[-1]))
    with open(file, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')


def generate_singal(args, model_name, test_case):
    result_path = args.result_path
    model_inference = inference_map[model_name](
        model_name=model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_dialog_turns=args.max_dialog_turns,
        user_model=args.user_model,
        language=args.language,
    )

    if 'agent' in test_case['id']:
        id, question, functions = test_case['id'], test_case['question'], test_case['function']
        if isinstance(functions, (dict, str)):
            functions = [functions]
        time = ''
        profile = ''
        result, process_list = model_inference.inference(question, functions, time, profile, test_case, id)
        result_to_write = {'id': id, 'result': result, 'process': process_list}
        model_inference.write_result(result_to_write, model_name, result_path)

    elif 'preference' in test_case['id']:
        id, question, functions, profile = (
            test_case['id'],
            test_case['question'],
            test_case['function'],
            test_case['profile'],
        )
        time = ''
        if isinstance(functions, (dict, str)):
            functions = [functions]

        result = model_inference.inference(question, functions, time, profile, test_case, id)

        result_to_write = {
            'id': id,
            'result': result,
        }
        model_inference.write_result(result_to_write, model_name, result_path)

    else:
        id, question, functions, time = (
            test_case['id'],
            test_case['question'],
            test_case['function'],
            test_case['time'],
        )
        profile = ''
        if isinstance(functions, (dict, str)):
            functions = [functions]

        result = model_inference.inference(question, functions, time, profile, test_case, id)

        result_to_write = {
            'id': id,
            'result': result,
        }
        model_inference.write_result(result_to_write, model_name, result_path)


def generate_results(args, model_name, test_case, completed_id_set):
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for test_case in test_cases_total:
            if test_case['id'] not in completed_id_set:
                future = executor.submit(generate_singal, args, model_name, test_case)
                futures.append(future)

        with tqdm(total=len(futures), desc='Processing Tasks', leave=True) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f'Task raised an exception: {e}')
                    raise
        print('All tasks have been completed.')


if __name__ == '__main__':
    args = get_args()

    if type(args.model) is not list:
        args.model = [args.model]
    if type(args.category) is not list:
        args.category = [args.category]

    paths = {
        'zh': {'data_path': './data_all/data_zh/', 'result_path': './result_all/result_zh/'},
        'en': {'data_path': './data_all/data_en/', 'result_path': './result_all/result_en/'},
    }

    data_path = paths[args.language]['data_path']
    result_path = paths[args.language]['result_path']
    args.result_path = result_path

    test_names = {test_name for category in args.category for test_name in ACE_DATA_CATEGORY[category]}
    test_files = [f'data_{test_name}.json' for test_name in test_names]

    for model_name in args.model:
        folder_path = os.path.join(result_path, model_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        completed_id_set = set()
        for file in test_names:
            file_name = f'data_{file}_result.json'
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, encoding='utf-8') as f:
                    for line in f:
                        line_data = json.loads(line)
                        completed_id_set.add(line_data['id'])
        test_cases_total = load_test_cases(data_path, test_files)

        if len(test_cases_total) > 0:
            generate_results(args, model_name, test_cases_total, completed_id_set)

        for file in test_names:
            file_name = f'data_{file}_result.json'
            file_path = os.path.join(folder_path, file_name)
            sort_json(file_path)

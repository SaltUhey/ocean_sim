import sys

def print_progress(iteration, total, prefix='PE Calculation:', length=40, fill='█'):
    """
    コンソールの同じ行を上書きして進捗バーを表示する関数
    """
    if total == 0:
        return
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% ({iteration}/{total} frames)')
    sys.stdout.flush()
    
    if iteration == total:
        print()
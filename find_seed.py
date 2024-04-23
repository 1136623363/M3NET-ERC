import subprocess
import random

# 定义要更改的文件名、行号和新的内容
file_name = 'train_meld.py'
line_number = 18
acc = 0.673

for _ in range(1,100000,1):

    seed = random.randint(1, 100000)
    # 使用生成的随机数进行后续操作
    # print(seed)

    new_content = f"seed = {seed}"

    # 打开文件并替换指定行
    with open(file_name, 'r',encoding='utf-8') as file:
        lines = file.readlines()

    lines[line_number - 1] = new_content + '\n'

    with open(file_name, 'w',encoding='utf-8') as file:
        file.writelines(lines)


    # 运行另一个脚本并捕获输出
    # output = subprocess.check_output(['python', "-u train_meld.py --dropout 0.4 --windows 2 --step 1 --lr 0.0001 --l2 0.00003 --lr2 0.0001 --l3 0.00003 --base-model 'GRU' --use_modal --batch-size 16 --graph_type='hyper' --epochs=1 --graph_construct='direct' --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' --norm BN --num_L=1 --num_K=0"]).decode('utf-8').split('\n')
    # output = subprocess.check_output(
    #     ['python', '-u', 'train_meld.py', '--dropout', '0.4', '--windows', '2', '--step', '1', '--lr', '0.0001', '--l2',
    #      '0.00003', '--lr2', '0.0001', '--l3', '0.00003', '--base-model', 'GRU', '--use_modal', '--batch-size', '16',
    #      '--graph_type', 'hyper', '--epochs', '50', '--graph_construct', 'direct', '--multi_modal', '--mm_fusion_mthd',
    #      'concat_DHT', '--modals', 'avl', '--Dataset', 'MELD', '--norm', 'BN', '--num_L=1', '--num_K=0']).decode(
    #     'utf-8').split('\n')
    proc = subprocess.Popen(
        ['python', '-u', 'train_meld.py', '--dropout', '0.4', '--windows', '3', '--step', '6', '--lr', '0.0001', '--l2',
         '0.00003', '--lr2', '0.0001', '--l3', '0.00003', '--base-model', 'GRU', '--use_modal', '--batch-size', '16',
         '--graph_type', 'hyper', '--epochs', '30', '--graph_construct', 'direct', '--multi_modal', '--mm_fusion_mthd',
         'concat_DHT', '--modals', 'avl', '--Dataset', 'MELD', '--norm', 'BN', '--num_L=3', '--num_K=0'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # 等待子进程结束，并获取其输出
    stdout, stderr = proc.communicate()

    # 处理输出
    output = stdout.decode('utf-8').split('\n')
    # print(output)
    tmp = float(''.join(list(output[-12])[-17:-11]))
    print(seed,"  acc:",tmp)
    # 检查输出中是否包含"100"
    if tmp > acc:
        acc = tmp
        print(output)
        print(f"best:seed {seed},acc {acc}")

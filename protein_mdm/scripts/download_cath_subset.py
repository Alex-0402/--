"""
CATH S40 数据集子集下载脚本

自动从 CATH 数据库下载 S40 非冗余数据集的子集（用于快速开发和测试）。

使用方法:
    # 下载默认 50 个蛋白质
    python scripts/download_cath_subset.py
    
    # 下载指定数量的蛋白质
    python scripts/download_cath_subset.py --limit 100
    
    # 指定输出目录
    python scripts/download_cath_subset.py --limit 50 --output_dir data/pdb_files
"""

import argparse
import os
import sys
import random
import time
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen, urlretrieve
from urllib.error import URLError, HTTPError

import requests
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# CATH 官方 URL（尝试多个可能的 URL）
CATH_S40_LIST_URLS = [
    "https://www.cathdb.info/download/by_release/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.list",
    "http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.list",
    "https://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.list",
]

# RCSB PDB 下载 URL 模板
RCSB_PDB_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"

# HTTP 请求头（模拟浏览器）
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}


def download_file(url: str, output_path: str, timeout: int = 30, headers: Optional[dict] = None) -> bool:
    """
    下载文件到指定路径
    
    Args:
        url: 文件 URL
        output_path: 输出文件路径
        timeout: 超时时间（秒）
        headers: HTTP 请求头（可选）
    
    Returns:
        是否下载成功
    """
    try:
        if headers is None:
            headers = HEADERS
        
        response = requests.get(url, timeout=timeout, stream=True, headers=headers)
        response.raise_for_status()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 写入文件
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except (requests.RequestException, IOError) as e:
        print(f"   ⚠️  下载失败: {e}")
        return False


def download_cath_list(output_path: str) -> bool:
    """
    从 CATH 官方下载 S40 非冗余列表文件
    
    尝试多个可能的 URL，直到成功为止
    
    Args:
        output_path: 输出文件路径
    
    Returns:
        是否下载成功
    """
    print(f"正在下载 CATH S40 列表文件...")
    print(f"  保存到: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 尝试多个 URL
    for i, url in enumerate(CATH_S40_LIST_URLS, 1):
        print(f"  尝试 URL {i}/{len(CATH_S40_LIST_URLS)}: {url}")
        success = download_file(url, output_path, headers=HEADERS)
        
        if success:
            # 验证文件内容（至少应该有一些行）
            try:
                with open(output_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                    if len(lines) > 0:
                        print(f"  ✅ CATH S40 列表文件下载成功（{len(lines)} 条记录）")
                        return True
                    else:
                        print(f"  ⚠️  文件下载成功但内容为空，尝试下一个 URL...")
                        os.remove(output_path)
            except Exception as e:
                print(f"  ⚠️  文件验证失败: {e}，尝试下一个 URL...")
                if os.path.exists(output_path):
                    os.remove(output_path)
        else:
            print(f"  ⚠️  下载失败，尝试下一个 URL...")
    
    print(f"  ❌ 所有 URL 都下载失败")
    print(f"\n提示：")
    print(f"  1. 可以手动从 CATH 网站下载列表文件")
    print(f"  2. 访问: https://www.cathdb.info/")
    print(f"  3. 或使用 --list_file 参数指定已有的列表文件")
    
    return False


def parse_cath_list(file_path: str) -> List[str]:
    """
    解析 CATH S40 列表文件
    
    CATH ID 格式通常是 7 位字符，例如：
    - 1oaiA00: 前4位是 PDB ID (1oai)，第5位是 Chain ID (A)
    - 或者直接是 PDB ID
    
    Args:
        file_path: 列表文件路径
    
    Returns:
        CATH ID 列表
    """
    cath_ids = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # CATH ID 可能是 7 位字符（如 1oaiA00）或 4 位 PDB ID
                # 提取前 4 位作为 PDB ID
                if len(line) >= 4:
                    cath_ids.append(line)
    except FileNotFoundError:
        print(f"  ❌ 文件不存在: {file_path}")
    except Exception as e:
        print(f"  ❌ 解析文件时出错: {e}")
    
    return cath_ids


def extract_pdb_id(cath_id: str) -> str:
    """
    从 CATH ID 中提取 PDB ID
    
    CATH ID 格式：
    - 7 位字符：1oaiA00 -> PDB ID: 1oai
    - 4 位字符：1oai -> PDB ID: 1oai
    
    Args:
        cath_id: CATH ID
    
    Returns:
        PDB ID（小写）
    """
    # 取前 4 位作为 PDB ID
    pdb_id = cath_id[:4].lower()
    return pdb_id


def download_pdb_file(pdb_id: str, output_dir: str) -> bool:
    """
    从 RCSB PDB 下载 PDB 文件
    
    Args:
        pdb_id: PDB ID（4 位字符，小写）
        output_dir: 输出目录
    
    Returns:
        是否下载成功
    """
    # 构造 URL（RCSB PDB 使用小写 PDB ID）
    url = RCSB_PDB_URL_TEMPLATE.format(pdb_id=pdb_id.lower())
    
    # 输出文件路径
    output_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    
    # 如果文件已存在，跳过
    if os.path.exists(output_path):
        return True
    
    # 下载文件（RCSB PDB 通常不需要特殊 headers，但为了兼容性还是传递）
    success = download_file(url, output_path, headers=HEADERS)
    
    if success:
        # 验证文件是否有效（至少应该有一些内容）
        try:
            if os.path.getsize(output_path) < 100:
                os.remove(output_path)
                return False
        except OSError:
            return False
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="下载 CATH S40 数据集的子集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载默认 50 个蛋白质
  python scripts/download_cath_subset.py
  
  # 下载 100 个蛋白质
  python scripts/download_cath_subset.py --limit 100
  
  # 指定输出目录
  python scripts/download_cath_subset.py --limit 50 --output_dir data/pdb_files
  
  # 使用随机采样
  python scripts/download_cath_subset.py --limit 50 --random
        """
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="下载的蛋白质数量（默认: 50）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="raw_data",
        help="PDB 文件输出目录（默认: raw_data）"
    )
    
    parser.add_argument(
        "--list_file",
        type=str,
        default=None,
        help="CATH 列表文件路径（如果已存在，跳过下载）"
    )
    
    parser.add_argument(
        "--random",
        action="store_true",
        help="随机采样（默认: 取前 N 个）"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）"
    )
    
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="跳过已存在的 PDB 文件"
    )
    
    parser.add_argument(
        "--generate_sample_list",
        action="store_true",
        help="如果下载失败，生成一个示例列表文件（用于测试）"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("CATH S40 数据集子集下载工具")
    print("="*70)
    print(f"下载数量: {args.limit}")
    print(f"输出目录: {args.output_dir}")
    print(f"随机采样: {args.random}")
    if args.random:
        print(f"随机种子: {args.seed}")
    print("="*70)
    
    # 1. 下载或加载 CATH 列表文件
    if args.list_file and os.path.exists(args.list_file):
        list_file_path = args.list_file
        print(f"\n使用现有的列表文件: {list_file_path}")
    else:
        # 默认保存到 data/meta/cath_s40_list.txt
        list_file_path = os.path.join(project_root, "data", "meta", "cath_s40_list.txt")
        
        if os.path.exists(list_file_path):
            print(f"\n列表文件已存在: {list_file_path}")
            print("  跳过下载（如需重新下载，请删除该文件）")
        else:
            print(f"\n1. 下载 CATH S40 列表文件...")
            if not download_cath_list(list_file_path):
                if args.generate_sample_list:
                    print(f"\n  生成示例列表文件（用于测试）...")
                    # 生成一些常见的 PDB ID 作为示例
                    sample_ids = [
                        "1ubqA00", "1crnA00", "1oaiA00", "2lyzA00", "3ptnA00",
                        "1a2bA00", "1btaA00", "1cspA00", "1dktA00", "1eiyA00",
                        "1fcaA00", "1gflA00", "1hivA00", "1igdA00", "1jbeA00",
                        "1k2cA00", "1l2yA00", "1mboA00", "1nlsA00", "1opdA00",
                        "1pdoA00", "1qysA00", "1r69A00", "1sapA00", "1tigA00",
                        "1ubqA00", "1vlsA00", "1wbaA00", "1xysA00", "1yccA00",
                        "1zddA00", "2abdA00", "2bopA00", "2cbaA00", "2driA00",
                        "2eiyA00", "2fcaA00", "2gflA00", "2hivA00", "2igdA00",
                        "2jbeA00", "2k2cA00", "2l2yA00", "2mboA00", "2nlsA00",
                        "2opdA00", "2pdoA00", "2qysA00", "2r69A00", "2sapA00"
                    ]
                    os.makedirs(os.path.dirname(list_file_path), exist_ok=True)
                    with open(list_file_path, 'w') as f:
                        f.write('\n'.join(sample_ids))
                    print(f"  ✅ 已生成示例列表文件，包含 {len(sample_ids)} 个示例 ID")
                    print(f"  ⚠️  注意：这是示例数据，不是真实的 CATH S40 列表")
                else:
                    print("  ❌ 无法下载列表文件")
                    print("\n提示：")
                    print("  1. 可以手动从 CATH 网站下载列表文件")
                    print("  2. 访问: https://www.cathdb.info/")
                    print("  3. 或使用 --list_file 参数指定已有的列表文件")
                    print("  4. 或使用 --generate_sample_list 生成示例列表（用于测试）")
                    sys.exit(1)
    
    # 2. 解析列表文件
    print(f"\n2. 解析列表文件...")
    cath_ids = parse_cath_list(list_file_path)
    
    if len(cath_ids) == 0:
        print("  ❌ 列表文件为空或解析失败")
        sys.exit(1)
    
    print(f"  找到 {len(cath_ids)} 个 CATH ID")
    
    # 3. 采样子集
    print(f"\n3. 选择子集...")
    if args.random:
        random.seed(args.seed)
        selected_ids = random.sample(cath_ids, min(args.limit, len(cath_ids)))
        print(f"  随机选择了 {len(selected_ids)} 个 ID（种子: {args.seed}）")
    else:
        selected_ids = cath_ids[:args.limit]
        print(f"  选择了前 {len(selected_ids)} 个 ID")
    
    # 提取唯一的 PDB ID（可能有多个 CATH ID 对应同一个 PDB）
    pdb_ids = []
    seen_pdb_ids = set()
    for cath_id in selected_ids:
        pdb_id = extract_pdb_id(cath_id)
        if pdb_id not in seen_pdb_ids:
            pdb_ids.append(pdb_id)
            seen_pdb_ids.add(pdb_id)
    
    print(f"  提取到 {len(pdb_ids)} 个唯一的 PDB ID")
    
    # 4. 创建输出目录
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n4. 下载 PDB 文件到: {output_dir}")
    
    # 5. 下载 PDB 文件
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    # 使用 tqdm 显示进度
    with tqdm(total=len(pdb_ids), desc="下载进度", unit="文件") as pbar:
        for pdb_id in pdb_ids:
            output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
            
            # 检查文件是否已存在
            if args.skip_existing and os.path.exists(output_path):
                skipped_count += 1
                pbar.set_postfix({"成功": success_count, "失败": failed_count, "跳过": skipped_count})
                pbar.update(1)
                continue
            
            # 下载文件
            if download_pdb_file(pdb_id, output_dir):
                success_count += 1
            else:
                failed_count += 1
            
            pbar.set_postfix({"成功": success_count, "失败": failed_count, "跳过": skipped_count})
            pbar.update(1)
            
            # 避免请求过快，添加小延迟
            time.sleep(0.1)
    
    # 6. 输出结果
    print("\n" + "="*70)
    print("下载完成！")
    print("="*70)
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    if args.skip_existing:
        print(f"跳过: {skipped_count}")
    print(f"总计: {len(pdb_ids)}")
    print(f"\n文件保存在: {output_dir}")
    print("="*70)
    
    # 7. 生成文件列表（可选）
    if success_count > 0:
        list_output_path = os.path.join(output_dir, "downloaded_files.txt")
        with open(list_output_path, 'w') as f:
            for pdb_id in pdb_ids:
                pdb_file = os.path.join(output_dir, f"{pdb_id}.pdb")
                if os.path.exists(pdb_file):
                    f.write(f"{pdb_id}\n")
        print(f"\n已保存文件列表到: {list_output_path}")


if __name__ == "__main__":
    main()

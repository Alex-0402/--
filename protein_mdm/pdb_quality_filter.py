#!/usr/bin/env python3
"""
PDB 数据质量过滤工具 (PDB Quality Filter)

用于过滤和清洗用于侧链打包（Side-chain Packing）训练的 PDB/mmCIF 数据集。
筛选维度：
1. 晶体学分辨率 (Resolution) <= 2.5 Å (可配置)
2. 序列长度 (Length) 介于限制之间 (例如 30 ~ 2048)
3. 链的连续性 (CA-CA Distance) <= 4.5 Å (排除断裂肽键)
4. 排除含有大量非规范氨基酸或背骨残缺的脏数据
"""

import os
import argparse
import warnings
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, PDBExceptions
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import multiprocessing
from tqdm import tqdm

# 屏蔽烦人的 PDB 解析警告
warnings.simplefilter('ignore', PDBExceptions.PDBConstructionWarning)

def get_resolution(file_path, is_cif):
    """提取结构的分辨率，如果未记录则返回 None"""
    try:
        if is_cif:
            mmcif_dict = MMCIF2Dict(file_path)
            # 晶体学分辨率
            if '_refine.ls_d_res_high' in mmcif_dict:
                res = mmcif_dict['_refine.ls_d_res_high'][0]
                if res not in ('.', '?'):
                    return float(res)
            # 冷冻电镜分辨率
            if '_em_3d_reconstruction.resolution' in mmcif_dict:
                res = mmcif_dict['_em_3d_reconstruction.resolution'][0]
                if res not in ('.', '?'):
                    return float(res)
        else:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith('REMARK   2 RESOLUTION.'):
                        try:
                            # REMARK   2 RESOLUTION. 2.10 ANGSTROMS.
                            parts = line.split()
                            for p in parts:
                                try:
                                    return float(p)
                                except ValueError:
                                    continue
                        except:
                            pass
        return None
    except Exception:
        return None


def process_single_pdb(args):
    """多进程处理的单核工作流"""
    file_path, opts = args
    filename = os.path.basename(file_path)
    file_id = filename.split('.')[0]
    is_cif = file_path.endswith('.cif') or file_path.endswith('.CIF')
    
    # 1. 检查分辨率
    res = get_resolution(file_path, is_cif)
    if res is None and not opts['allow_missing_res']:
        return file_id, False, "Missing_Resolution"
    if res is not None and res > opts['max_resolution']:
        return file_id, False, f"Low_Resolution({res:.2f})"

    try:
        # 解析器
        parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
        structure = parser.get_structure('protein', file_path)
    except Exception as e:
        return file_id, False, f"Parse_Error"
    
    # 2. 获取有效氨基酸序列
    valid_residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # 排除水分子、杂原子 (Hetero atoms)
                if residue.id[0] != ' ':
                    continue
                # 检查主骨架原子是否完整
                if 'N' in residue and 'CA' in residue and 'C' in residue:
                    valid_residues.append(residue)
        break # 仅看第一个Model (NMR会有多个)
        
    num_res = len(valid_residues)
    
    # 3. 检查长度
    if num_res < opts['min_length']:
        return file_id, False, f"Too_Short({num_res})"
    if num_res > opts['max_length']:
        return file_id, False, f"Too_Long({num_res})"
        
    # 4. 检查多肽链断裂 (CA-CA 强几何约束)
    # 计算相邻残基 CA 原子的欧氏距离
    breaks_count = 0
    for i in range(len(valid_residues) - 1):
        # 必须是同一条链
        if valid_residues[i].get_parent().id == valid_residues[i+1].get_parent().id:
            ca_dist = valid_residues[i]['CA'] - valid_residues[i+1]['CA']
            # 标准多肽键连的 CA-CA 距离在 3.8 Å 左右，如果大于 4.5 认为物理链断裂
            if ca_dist > opts['max_ca_dist']:
                breaks_count += 1
                
    if breaks_count > opts['max_breaks']:
        return file_id, False, f"Too_Many_Chain_Breaks({breaks_count})"
        
    return file_id, True, "Pass"

def main():
    parser = argparse.ArgumentParser(description="PDB/mmCIF 数据纯化过滤脚本")
    parser.add_argument("--raw_data_dir", type=str, default="raw_data", help="包含原始PDB的文件夹")
    parser.add_argument("--output_file", type=str, default="data/high_quality_pdbs.txt", help="合格样本列表保存路径")
    parser.add_argument("--max_resolution", type=float, default=2.5, help="最高允许的分辨率(Å)")
    parser.add_argument("--allow_missing_res", action="store_true", help="是否允许没有分辨率信息的结构(如NMR)")
    parser.add_argument("--min_length", type=int, default=40, help="最小残基数")
    parser.add_argument("--max_length", type=int, default=2048, help="最大残基数 (防OOM)")
    parser.add_argument("--max_ca_dist", type=float, default=4.5, help="CA-CA相连断裂的容忍阈值(Å)")
    parser.add_argument("--max_breaks", type=int, default=2, help="允许的链断裂最大次数")
    parser.add_argument("--num_workers", type=int, default=8, help="多进程并发数")
    args = parser.parse_args()

    if not os.path.exists(args.raw_data_dir):
        print(f"Error: 找不到数据目录 {args.raw_data_dir}")
        return

    # 收集文件
    all_files = []
    for ext in ["*.pdb", "*.CIF", "*.cif", "*.PDB", "*.ent"]:
        # 支持一级及二级子目录
        for root, _, files in os.walk(args.raw_data_dir):
            for f in files:
                if f.endswith(tuple(ext.replace('*','').split())):
                    all_files.append(os.path.join(root, f))
    
    # 去重
    all_files = list(set(all_files))
    print(f"✅ 在 {args.raw_data_dir} 中找到 {len(all_files)} 个结构文件，准备开始筛选...")
    
    opts = {
        'max_resolution': args.max_resolution,
        'allow_missing_res': args.allow_missing_res,
        'min_length': args.min_length,
        'max_length': args.max_length,
        'max_ca_dist': args.max_ca_dist,
        'max_breaks': args.max_breaks
    }

    work_items = [(f, opts) for f in all_files]
    
    passed_ids = []
    reject_reasons = {}
    
    with multiprocessing.Pool(args.num_workers) as pool:
        for file_id, is_pass, reason in tqdm(pool.imap_unordered(process_single_pdb, work_items), total=len(work_items), desc="过滤结构"):
            if is_pass:
                passed_ids.append(file_id)
            else:
                reject_reasons[reason.split('(')[0]] = reject_reasons.get(reason.split('(')[0], 0) + 1

    # 保存合格名单
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for pid in sorted(passed_ids):
            f.write(f"{pid}\n")
            
    print("\n" + "="*50)
    print("🎯 PDB 过滤筛选完成 (Quality Filter Complete)")
    print("="*50)
    print(f"总文件数: {len(all_files)}")
    print(f"✅ 合格数: {len(passed_ids)} ({(len(passed_ids)/len(all_files)*100 if all_files else 0):.1f}%)")
    print(f"❌ 剔除数: {len(all_files) - len(passed_ids)}")
    print("\n--- 失败原因统计 (Top Reasons) ---")
    for r, count in sorted(reject_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f" - {r}: {count} 个")
    print(f"\n合格的 PDB IDs 已经保存至: {args.output_file}")
    print("你可以使用该列表代替你原本随机的全部列表来重新划定 train/val！")

if __name__ == '__main__':
    main()

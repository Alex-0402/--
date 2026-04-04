import urllib.request
import os
import time

def download_pdb(pdb_id, save_dir):
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    save_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(save_path):
        print(f"[{pdb_id}] Already exists, skipping.")
        return True
        
    try:
        # 添加 User-Agent 头，防止被 RCSB 拒绝
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(save_path, 'wb') as f:
                f.write(response.read())
        print(f"[{pdb_id}] Downloaded successfully.")
        return True
    except Exception as e:
        print(f"[{pdb_id}] Failed to download: {e}")
        return False

# 假设用户已经手动下载了 list 并且存为 cath42_test_ids.txt，每行一个 PDB ID（例如 "1abc"）
def main():
    target_dir = "./raw_data/cath42_test_pdbs"
    os.makedirs(target_dir, exist_ok=True)
    
    id_list_path = "cath42_test_ids.txt"
    if not os.path.exists(id_list_path):
        print(f"❌ 找不到 {id_list_path}！请手动将 CATH 4.2 的测试集 ID 列表保存为该文件。")
        return
        
    with open(id_list_path, "r") as f:
        # 清理每行的空白字符，取前4个字符作为 PDB ID (忽略 chain 标识符，如 1abcA -> 1abc)
        pdb_ids = list(set([line.strip()[:4] for line in f if line.strip()]))
        
    print(f"🔍 从列表中解析出 {len(pdb_ids)} 个唯一的 PDB ID。开始下载...")
    
    success_count = 0
    for i, pdb_id in enumerate(pdb_ids):
        print(f"[{i+1}/{len(pdb_ids)}] ", end="")
        if download_pdb(pdb_id, target_dir):
            success_count += 1
        time.sleep(0.5)  # 礼貌性延迟，避免被服务器拉黑
        
    print(f"✅ 下载完成！成功: {success_count}/{len(pdb_ids)}。数据保存在 {target_dir}")

if __name__ == '__main__':
    main()

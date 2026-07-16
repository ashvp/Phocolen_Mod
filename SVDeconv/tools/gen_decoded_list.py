import os

def main():
    list_dir = "/4tb_hdd/Ashwin/lenslessFlatnetEndoNeRF/text_files"
    splits = ["train", "val"]
    
    print(f"Generating decoded capture list files in: {list_dir}")
    
    for split in splits:
        target_list_path = os.path.join(list_dir, f"{split}_target.txt")
        decoded_list_path = os.path.join(list_dir, f"decoded_sim_captures_{split}.txt")
        
        if not os.path.exists(target_list_path):
            print(f"Error: Target list file not found at {target_list_path}")
            continue
            
        with open(target_list_path, 'r') as f:
            lines = f.readlines()
            
        with open(decoded_list_path, 'w') as f:
            for line in lines:
                filename = line.strip().split('/')[-1]
                # Write path relative to image_dir: decoded_sim_captures/filename
                f.write(f"decoded_sim_captures/{filename}\n")
                
        print(f"Successfully generated: {decoded_list_path} ({len(lines)} entries)")

if __name__ == "__main__":
    main()

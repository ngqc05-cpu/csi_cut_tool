import struct
import numpy as np
import pandas as pd
import os
import json

PACKET_COUNT = 1000
SEQ_MAX = 4096
DEVICE_CONFIG = {
    'esp': {
        'packet_size': 144,
    },
    'asus': {
        'packet_size': 1044,
    }
}
CONFIG_FILE = "csi_config.json"


def get_database_path() -> str:
    """Tự động lấy đường dẫn Database từ file cấu hình config.json"""
    config_data = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except Exception:
            pass

    saved_path = config_data.get("database_root", "")
    if saved_path and os.path.exists(saved_path):
        print(f"\n[*] Đã tải cấu hình Database gốc từ lần chạy trước:")
        print(f" -> {saved_path}")
        choice = input("Bấm [Enter] để sử dụng, hoặc gõ 'n' để đổi đường dẫn mới: ").strip().lower()
        if choice != 'n':
            return saved_path

    while True:
        print("\n" + "="*50)
        new_path = input(" [Cài đặt] Nhập đường dẫn thư mục Database gốc: ").strip().strip('"\'')
        if os.path.exists(new_path):
            config_data["database_root"] = new_path
            try:
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=4, ensure_ascii=False)
                print(" -> [Thành công] Đã lưu đường dẫn vĩnh viễn!")
            except Exception as e:
                print(f" -> [Cảnh báo] Không thể lưu file cấu hình: {e}")
            return new_path
        else:
            print(" -> [Lỗi] Đường dẫn không tồn tại. Vui lòng kiểm tra lại.")


def get_optional_input(prompt_text: str) -> str:
    val = input(prompt_text).strip()
    return val if val != "" else None


def search_and_select_folder(db_root: str) -> str:
    if not os.path.exists(db_root):
        print(f"[Lỗi] Không tìm thấy thư mục Database gốc: {db_root}")
        return None

    print("\n" + "="*50)
    print(" BỘ LỌC TÌM KIẾM DỮ LIỆU DATABASE")
    print(" (Mẹo: Bấm Enter để bỏ qua tiêu chí nếu không cần thiết)")
    print("="*50)

    room = get_optional_input(" -> Chỉ số phòng (room)      : ")
    setup = get_optional_input(" -> Chỉ số setup (setup)     : ")
    session = get_optional_input(" -> Chỉ số phiên (session)   : ")
    user_idx = get_optional_input(" -> Chỉ số người (user)      : ")
    pos = get_optional_input(" -> Chỉ số vị trí (pos)      : ")
    
    criteria = [room, setup, session, user_idx, pos]
    matched_folders = []
    
    for folder_name in os.listdir(db_root):
        folder_path = os.path.join(db_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        parts = folder_name.split('_')
        if len(parts) < 5:
            continue
            
        is_match = True
        for i in range(5):
            if criteria[i] is not None and criteria[i] != parts[i]:
                is_match = False
                break
        if is_match:
            matched_folders.append(folder_name)
            
    if not matched_folders:
        print("\n[Thất bại] Không tìm thấy folder nào khớp.")
        return None
        
    if len(matched_folders) == 1:
        print(f"\n[Thành công] Tìm thấy duy nhất 1 thư mục khớp: {matched_folders[0]}")
        return os.path.join(db_root, matched_folders[0])
        
    print(f"\n[Kết quả] Tìm thấy {len(matched_folders)} thư mục phù hợp:")
    for idx, f_name in enumerate(matched_folders):
        print(f"  {idx + 1}. {f_name}")
        
    while True:
        choice = input(f"\n -> Chọn số thứ tự thư mục muốn xử lý (1 - {len(matched_folders)}): ").strip()
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(matched_folders):
                return os.path.join(db_root, matched_folders[choice_idx])
            print(" [!] Lựa chọn nằm ngoài phạm vi danh sách.")
        except ValueError:
            print(" [!] Vui lòng nhập một số nguyên hợp lệ.")


def get_time_from_event(event_file: str, action_name: str, repeat_idx: int) -> int:
    if not os.path.exists(event_file):
        print(f"[Lỗi] không tìm thấy file event: {event_file}")
        return None
    try:
        df = pd.read_csv(event_file) if event_file.endswith('.csv') else pd.read_excel(event_file)
        required_columns = ['action_name', 'repeat_index', 'start_unix_us']
        for col in required_columns:
            if col not in df.columns:
                print(f"[Lỗi] File event thiếu cột: '{col}'")
                return None
                
        condition = (df['action_name'] == action_name) & (df['repeat_index'] == repeat_idx)
        filtered_df = df[condition]
        if filtered_df.empty:
            print(f"[Cảnh báo] Không tìm thấy '{action_name}' lần lặp {repeat_idx}.")
            return None
            
        return int(float(filtered_df['start_unix_us'].iloc[0]))
    except Exception as e:
        print(f"[Lỗi xử lý Dataframe] {e}")
        return None


def find_first_packet_fast(file_path: str, target_time: int, packet_size: int, excel_output: str) -> tuple:
    if not os.path.exists(file_path):
        print(f"[Lỗi] Không tìm thấy file: {file_path}")
        return None, None

    file_size = os.path.getsize(file_path)
    total_packets = file_size // packet_size
    if total_packets == 0: return None, None

    print(f"[{os.path.basename(file_path)}] Đang quét {total_packets} gói tin...")
    best_seq, best_ts = None, None
    best_index = -1
    
    with open(file_path, 'rb') as f:
        low, high = 0, total_packets - 1
        while low <= high:
            mid = (low + high) // 2
            f.seek(mid * packet_size)
            header_data = f.read(10)
            if len(header_data) < 10: break
                
            seq, ts = struct.unpack('<HQ', header_data)
            if ts >= target_time:
                best_seq, best_ts = seq, ts
                best_index = mid
                high = mid - 1
            else:
                low = mid + 1
                
    if best_index != -1:
        try:
            pd.DataFrame({
                "Sequence": [best_seq], "Timestamp": [best_ts],
                "Target_Time_Input": [target_time], "Time_Difference": [best_ts - target_time]
            }).to_excel(excel_output, index=False)
            print(f" -> Điểm neo chuẩn xác -> Seq: {best_seq} | TS: {best_ts}")
        except Exception as e:
            print(f" -> [Cảnh báo] Không thể ghi file Excel log: {e}")
        return best_seq, best_ts
            
    print(f" -> [Thất bại] Không có gói tin nào phù hợp.")
    return None, None


def find_anchor_offset(file_path: str, target_seq: int, target_ts: int, packet_size: int, tolerance: int = 1000000) -> int:
    if not os.path.exists(file_path): return None
    file_size = os.path.getsize(file_path)
    total_packets = file_size // packet_size
    if total_packets == 0: return None

    search_time = target_ts - 600000 
    low, high = 0, total_packets - 1
    start_index = 0

    with open(file_path, 'rb') as f:
        while low <= high:
            mid = (low + high) // 2
            f.seek(mid * packet_size)
            header = f.read(10)
            if not header: break
            _, ts = struct.unpack('<HQ', header)
            if ts >= search_time:
                start_index = mid
                high = mid - 1
            else:
                low = mid + 1

    with open(file_path, 'rb') as f:
        f.seek(start_index * packet_size)
        while True:
            current_offset = f.tell()
            packet = f.read(packet_size)
            if not packet or len(packet) < 16: break
                
            seq, ts = struct.unpack('<HQ', packet[:10])
            if ts > target_ts + tolerance: return None 
            
            seq_diff = (seq - target_seq) % SEQ_MAX
            if seq_diff == 0 and abs(ts - target_ts) <= tolerance:
                return current_offset 
            if ts >= target_ts and 0 < seq_diff <= 100:
                return current_offset
    return None 


def cut_and_pad_1000(file_in: str, start_offset: int, target_seq: int, packet_size: int, out_filename: str = None) -> bytearray:
    """Cắt và đồng bộ gói tin. Trả về vùng đệm bytearray, chỉ lưu xuống đĩa nếu out_filename được cung cấp."""
    output_data = bytearray()
    packets_collected = 0
    expected_seq = target_seq

    with open(file_in, 'rb') as f:
        f.seek(start_offset)
        while packets_collected < PACKET_COUNT:
            current_pos = f.tell()
            packet = f.read(packet_size)

            if not packet or len(packet) < packet_size:
                output_data.extend(b'\x00' * packet_size)
                expected_seq = (expected_seq + 1) % SEQ_MAX
                packets_collected += 1
                continue

            seq, _ = struct.unpack('<HQ', packet[:10])

            if seq == expected_seq:
                output_data.extend(packet)
                expected_seq = (expected_seq + 1) % SEQ_MAX
                packets_collected += 1
            else:
                seq_diff = (seq - expected_seq) % SEQ_MAX
                if seq_diff > 100:
                    print(f"[{os.path.basename(file_in)}] Cảnh báo: Nhảy quãng dữ liệu ({seq_diff} gói).")
                output_data.extend(b'\x00' * packet_size)
                expected_seq = (expected_seq + 1) % SEQ_MAX
                packets_collected += 1
                f.seek(current_pos)

    if out_filename:
        with open(out_filename, 'wb') as f_out:
            f_out.write(output_data)
        print(f"[Thành công] Đã cắt và lưu 1000 gói -> {out_filename}")
    else:
        print(f"[RAM Process] Đã đệm và sửa lỗi 1000 gói thành công trên RAM.")

    return output_data


def sync_and_cut_3_files(file1, file2, file3, target_seq: int, target_ts: int, packet_size: int, dev_type: str, base_out_name: str, output_dir: str, is_save: bool) -> list:
    """Điều phối đồng bộ 3 Rx. Trả về danh sách 3 vùng đệm dữ liệu (buffers) trên RAM."""
    files = [file1, file2, file3]
    buffers = []
    
    for i, file_path in enumerate(files):
        rx_folder = f"{dev_type}{i+1}" 
        out_name = os.path.join(output_dir, rx_folder, base_out_name) if is_save else None
        
        print(f"\nĐang xử lý {rx_folder} ({os.path.basename(file_path)})...")
        anchor_offset = find_anchor_offset(file_path, target_seq, target_ts, packet_size)
        
        if anchor_offset is not None:
            buf = cut_and_pad_1000(file_path, anchor_offset, target_seq, packet_size, out_name)
            buffers.append(buf)
        else:
            print(f"[Thất bại] Không tìm thấy neo cho {rx_folder}. Tạo mảng byte 00 trống.")
            zero_data = b'\x00' * packet_size * PACKET_COUNT
            if is_save:
                with open(out_name, 'wb') as f_out:
                    f_out.write(zero_data)
            buffers.append(zero_data)
            
    return buffers


def extract_csi_matrix(data_inputs: list, dev_type: str, from_buffer: bool = False) -> list:
    """Trích xuất ma trận CSI từ danh sách File Path HOẶC danh sách Vùng đệm RAM (Buffer)."""
    results = []

    for rx_idx, input_source in enumerate(data_inputs):
        print(f"Đang giải mã ma trận ({dev_type.upper()}) cho Rx{rx_idx + 1}...")
        
        if dev_type == 'esp':
            esp_dtype = np.dtype([
                ('seq', '<u2'), ('timestamp', '<u8'), ('channel', '<u2'),
                ('agc', 'u1'), ('fft', 'u1'), ('noise', 'i1'), ('rssi', 'i1'),
                ('payload', 'i1', (128,))
            ])
            # Chọn nguồn đọc dữ liệu thích hợp
            data = np.frombuffer(input_source, dtype=esp_dtype) if from_buffer else np.fromfile(input_source, dtype=esp_dtype)
            
            Q_float = data['payload'][:, 0::2].astype(np.float32)
            I_float = data['payload'][:, 1::2].astype(np.float32)
            amplitude = np.sqrt(I_float**2 + Q_float**2)
            phase = np.arctan2(Q_float, I_float)
            
            results.append({
                'rx_index': rx_idx, 'timestamp': data['timestamp'], 
                'amplitude': amplitude, 'phase': phase
            })

        elif dev_type == 'asus':
            asus_dtype = np.dtype([
                ('seq', '<u2'), ('timestamp', '<u8'), ('channel', '<u2'),
                ('agc_gain', 'u1', (4,)), ('rssi', 'i1', (4,)),
                ('payload', '<u4', (256,))
            ])
            data = np.frombuffer(input_source, dtype=asus_dtype) if from_buffer else np.fromfile(input_source, dtype=asus_dtype)
            
            csi_raw = data['payload']
            s_q = (csi_raw >> 29) & 0x01
            m_q = (csi_raw >> 18) & 0x07ff
            e   = csi_raw & 0x3f
            
            s_i = (csi_raw >> 17) & 0x01
            m_i = (csi_raw >> 6) & 0x07ff
            
            sign_q = 1 - 2 * s_q 
            sign_i = 1 - 2 * s_i 
            exponent = 2.0 ** (e.astype(np.float32) - 127)
            
            Q_float = sign_q * (1 + m_q) * exponent
            I_float = sign_i * (1 + m_i) * exponent
            
            amplitude = np.sqrt(I_float**2 + Q_float**2).reshape(-1, 4, 64)
            phase = np.arctan2(Q_float, I_float).reshape(-1, 4, 64)
            
            results.append({
                'rx_index': rx_idx, 'timestamp': data['timestamp'], 
                'amplitude': amplitude, 'phase': phase
            })
            
    print("[Thành công] Đã trích xuất xong mảng đa chiều vào bộ nhớ RAM!")
    return results


def main():
    while True:
        dev_type = input("1. Chọn loại thiết bị (esp / asus): ").strip().lower()
        if dev_type in ['esp', 'asus']: break
        print("   -> Thiết bị không hợp lệ, vui lòng gõ 'esp' hoặc 'asus'.")
    
    pkt_size = DEVICE_CONFIG[dev_type]['packet_size']

    # 2. Tìm kiếm Database tự động thông minh
    db_root = get_database_path()
    path = search_and_select_folder(db_root)
    if path is None:
        input("\nNhấn Enter để thoát chương trình...")
        return

    # 3. Nhập tên hành động
    action = input("\n3. Nhập tên hành động (VD: cui, nga,...): ").strip()

    # 4. Nhập chỉ số lặp lại
    while True:
        try:
            repeat = int(input("4. Nhập chỉ số lần lặp thứ (VD: 1, 2, 3...): ").strip())
            break
        except ValueError:
            print("   -> Lỗi: Vui lòng nhập một số nguyên.")

    # 5. Xử lý lựa chọn cắt lưu file ổ đĩa
    is_save = False
    save_choice = input("5. Có muốn lưu file .bin đã cắt xuống ổ đĩa không? (y/N): ").strip().lower()
    if save_choice == 'y':
        is_save = True

    print("\n" + "-"*60)
    print(f" ĐANG CHẠY CHỨC NĂNG: EXARRAY (Lưu ổ đĩa: {is_save})")
    print("-"*60)

    # Khởi tạo thư mục tổng chứa kết quả (nếu có lưu)
    output_dir = "Cut_Data"
    if is_save:
        os.makedirs(output_dir, exist_ok=True)
        for d_type in ['esp', 'asus']:
            for i in range(1, 4):
                os.makedirs(os.path.join(output_dir, f"{d_type}{i}"), exist_ok=True)

    # Bóc tách tên file theo cú pháp
    folder_name = os.path.basename(os.path.normpath(path))
    parts = folder_name.split('_')
    if len(parts) >= 7:
        prefix_5 = "_".join(parts[:5])     
        time_suffix = "_".join(parts[-2:]) 
    else:
        prefix_5 = "1_1_1_1_1"
        time_suffix = "0000_000000"

    base_name = f"{prefix_5}_{repeat}_{action}_{time_suffix}"
    base_out_name = f"{base_name}.bin"
    print(f"[*] Tên file mẫu đồng bộ: {base_out_name}")

    event_file = os.path.join(path, "action_events.csv")
    f1 = os.path.join(path, f"raw_{dev_type}1.bin")
    f2 = os.path.join(path, f"raw_{dev_type}2.bin")
    f3 = os.path.join(path, f"raw_{dev_type}3.bin")

    time_input = get_time_from_event(event_file, action, repeat)
    if time_input is None:
        print("\n[Kết thúc] Dừng chương trình do không tìm thấy mốc thời gian.")
        input("Nhấn Enter để thoát...")
        return
        
    log_file_name = os.path.join(output_dir, f"{base_name}_log.xlsx") if is_save else "temp_log.xlsx"
    found_seq, found_ts = find_first_packet_fast(f1, time_input, pkt_size, log_file_name)
    
    if found_seq is None:
        print("\n[Kết thúc] Dừng chương trình do không tìm thấy điểm neo đồng bộ.")
        input("Nhấn Enter để thoát...")
        return

    # Gọi hàm cắt đồng bộ, nhận về mảng dữ liệu RAM trực tiếp 
    ram_buffers = sync_and_cut_3_files(f1, f2, f3, found_seq, found_ts, pkt_size, dev_type, base_out_name, output_dir, is_save=is_save)

    # Trích xuất thẳng ma trận vào bộ nhớ (Luôn kích hoạt vì mặc định là exarray)
    csi_matrices = extract_csi_matrix(ram_buffers, dev_type, from_buffer=True)
    
    if csi_matrices and csi_matrices[0] is not None:
        print(f"\n[Hoàn tất EXARRAY] Dữ liệu ma trận đã sẵn sàng trong RAM.")
        print(f" -> Kích thước mảng Rx1 (Amplitude): {csi_matrices[0]['amplitude'].shape}")

    print("\n[Thành công] Đã hoàn tất toàn bộ quy trình một cách tối ưu!")
    input("Nhấn Enter để thoát chương trình...")

if __name__ == "__main__":
    main()
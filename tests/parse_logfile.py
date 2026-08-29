import struct
import os

def parse_trace_file(fpath: str):
    with open(fpath, "rb") as fin:
        content = fin.read()

    marker = b"# TRACE\n"
    marker_pos = content.rfind(marker)

    if marker_pos == -1:
        raise ValueError("Marker '# TRACE' not found in file.")

    binary_data = content[marker_pos + len(marker) :]
    binary_data = binary_data.rstrip(b"\r\n")

    batch_size = 44
    if len(binary_data) % batch_size != 0:
        raise ValueError(f"Binary segment length ({len(binary_data)} bytes) is not a multiple of {batch_size} bytes.")

    num_batches = len(binary_data) // batch_size
    if num_batches == 0:
        raise ValueError("No binary batches found after marker.")

    # struct format:
    #   <  : Little-endian (use '>' for big-endian if needed)
    #   d  : 1 double (8 bytes)
    #   3I : 3 uint32_t (3 * 4 = 12 bytes)
    #   3d : 3 double (3 * 8 = 24 bytes)
    # Total: 8 + 12 + 24 = 44 bytes
    fmt = "<d3I3d"

    # Slice the last 44-byte batch
    last_batch_bytes = binary_data[-batch_size:]
    decoded_last_batch = struct.unpack(fmt, last_batch_bytes)

    return num_batches, decoded_last_batch


if __name__ == "__main__":
    root = "/home/vlad/Documents/SublimeMerge/dragonfly-scripts/server_mirror/uet-htsim/htsim/sim/datacenter/dragonfly_plus_experiments/"

    # files = ["logout_178783377255_0_0.dat", "logout_178783377255_2_0.dat", "logout_178783377255_1_0.dat", "logout_178783377255_7_0.dat", "logout_178783377255_8_0.dat", "logout_178783377255_6_0.dat", "logout_178783377255_4_0.dat", "logout_178783377255_3_0.dat", "logout_178783377255_5_0.dat"]
    # files = ["logout_178783377255_0_1_0.dat", "logout_178783377255_0_2_0.dat", "logout_178783377255_0_0_0.dat", "logout_178783377255_0_7_0.dat", "logout_178783377255_0_6_0.dat", "logout_178783377255_0_8_0.dat", "logout_178783377255_0_4_0.dat", "logout_178783377255_0_3_0.dat", "logout_178783377255_0_5_0.dat"]
    files = ["logout_178783377255_1_8_0.dat", "logout_178783377255_1_7_0.dat", "logout_178783377255_1_4_0.dat", "logout_178783377255_1_6_0.dat", "logout_178783377255_1_3_0.dat", "logout_178783377255_1_5_0.dat"]

    files.sort()

    for file in files:
        fpath = os.path.join(root, file)

        try:
            total_batches, last_batch = parse_trace_file(fpath)

            fct = last_batch[0] # in secunde.
            print(f"{file = }, fct(us) = {int(fct * 1e6)}")

            # print(f"\n--- Binary Trace Summary ---")
            # print(f"Total 44-byte batches found: {total_batches}")

            # print("\n--- Last Decoded Batch ---")
            # print(f"Double 1:   {last_batch[0]}")
            # print(f"UInt32 1-3: {last_batch[1:4]}")
            # print(f"Double 2-4: {last_batch[4:]}")

        except FileNotFoundError:
            print(f"Error: Could not find file '{fpath}'.")
        except ValueError as e:
            print(f"Parsing error: {e}")

"""
for (int i=0; i < numread; i++) {
    fwrite(&timeRec[i], sizeof(double), 1, logfile);
    fwrite(&typeRec[i], sizeof(uint32_t), 1, logfile);
    fwrite(&idRec[i],   sizeof(uint32_t), 1, logfile);
    fwrite(&evRec[i],   sizeof(uint32_t), 1, logfile);
    fwrite(&val1Rec[i], sizeof(double), 1, logfile);
    fwrite(&val2Rec[i], sizeof(double), 1, logfile);
    fwrite(&val3Rec[i], sizeof(double), 1, logfile);
}
"""

import pylidc as pl

# Ambil semua scan
scans = list(pl.query(pl.Scan))

# Index awal
current_idx = 0
total_scans = len(scans)


def show_scan(idx):
    scan = scans[idx]
    nodules = scan.cluster_annotations()

    print(f"\n=== Scan {idx+1}/{total_scans} ===")
    print(f"Patient ID: {scan.patient_id}")
    print(f"Study UID : {scan.study_instance_uid}")
    print(f"Series UID: {scan.series_instance_uid}")
    print(f"Jumlah nodules: {len(nodules)}")

    # Visualisasi pylidc
    scan.visualize(annotation_groups=nodules)


while True:
    show_scan(current_idx)

    print("\nPerintah:")
    print("[n] Next scan")
    print("[p] Previous scan")
    print("[j] Jump ke index")
    print("[id] Jump ke patient_id")
    print("[q] Quit")

    cmd = input("Masukkan perintah: ").strip().lower()

    if cmd == "n":
        current_idx = (current_idx + 1) % total_scans

    elif cmd == "p":
        current_idx = (current_idx - 1) % total_scans

    elif cmd == "j":
        try:
            idx = int(input(f"Masukkan index (1 - {total_scans}): "))
            if 1 <= idx <= total_scans:
                current_idx = idx - 1
            else:
                print("Index out of range!")
        except ValueError:
            print("Input tidak valid!")

    elif cmd == "id":
        pid = input("Masukkan patient_id (contoh: LIDC-IDRI-0001): ").strip()
        found = False
        for i, s in enumerate(scans):
            if s.patient_id == pid:
                current_idx = i
                found = True
                break
        if not found:
            print("Patient ID tidak ditemukan!")

    elif cmd == "q":
        print("Keluar...")
        break

    else:
        print("Perintah tidak dikenal!")
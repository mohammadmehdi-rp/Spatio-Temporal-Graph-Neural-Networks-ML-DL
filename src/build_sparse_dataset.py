import numpy as np
import argparse

def find_idx(feat_names, key):
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in feat_names]
    for i, n in enumerate(names):
        if n == key:
            return i
    return None

def load_txt(path, nodes):
    names = [x.strip() for x in open(path).read().splitlines() if x.strip()]
    idx = []
    for n in names:
        if n in nodes:
            idx.append(np.where(nodes == n)[0][0])
        else:
            print(f"[WARN] Sensor {n} not found in node list.")
    return np.array(idx, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_full", required=True, help="Path to dataset_full_v4.npz")
    ap.add_argument("--sensors_txt", required=True, help="File with selected interfaces (one per line)")
    ap.add_argument("--out", required=True, help="Output NPZ file")
    args = ap.parse_args()

    Z = np.load(args.npz_full, allow_pickle=True)
    nodes = Z["nodes"]
    X = Z["X"]   # [T, N, F]

    sensor_idx = load_txt(args.sensors_txt, nodes)
    is_sensor = np.zeros(nodes.shape[0], dtype=int)
    is_sensor[sensor_idx] = 1

    print("\n=== Building sparse dataset ===")
    print(f"Selected sensors: {nodes[sensor_idx]}")
    print(f"Sensors count: {len(sensor_idx)}")

    # Copy all original keys
    out = {k: Z[k] for k in Z.files}

    # Overwrite the is_sensor FEATURE channel in X (if it exists)
    feat_names = out.get("feat_names", None)
    X_new = X.copy()

    if feat_names is not None:
        ch_is = find_idx(feat_names, "is_sensor")
        if ch_is is not None:
            print(f"[INFO] Overwriting 'is_sensor' feature channel at index {ch_is}")
            # broadcast node-wise is_sensor mask to all time steps
            # X_new: [T, N, F]
            T, N, F = X_new.shape
            mask = is_sensor.astype(X_new.dtype)[None, :, None]   # [1, N, 1]
            X_new[:, :, ch_is:ch_is+1] = mask
        else:
            print("[WARN] 'is_sensor' feature not found in feat_names; leaving X unchanged.")
    else:
        print("[WARN] No feat_names in NPZ; leaving X unchanged.")

    out["X"] = X_new
    out["is_sensor"] = is_sensor
    out["sensors"] = nodes[is_sensor == 1]

    np.savez(args.out, **out)
    print(f"[OK] Saved → {args.out}")

if __name__ == "__main__":
    main()

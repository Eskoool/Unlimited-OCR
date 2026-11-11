"""OCR local con Unlimited-OCR-2 (modo Transformers, GPU).

Uso:
  .venv\\Scripts\\python.exe ocr.py documento.pdf            # todas las páginas
  .venv\\Scripts\\python.exe ocr.py documento.pdf --paginas 3
  .venv\\Scripts\\python.exe ocr.py foto.jpg

Salida: output/<nombre>/<nombre>.txt (una sección por página).
"""
import argparse, contextlib, io, os, time
import fitz, torch
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOT, "models", "Unlimited-OCR-2")


def pages_of(path, out_dir, limit):
    if not path.lower().endswith(".pdf"):
        return [path]
    doc = fitz.open(path)
    n = len(doc) if not limit else min(limit, len(doc))
    files = []
    for i in range(n):
        f = os.path.join(out_dir, f"page_{i + 1}.jpg")
        doc[i].get_pixmap(dpi=144).save(f)
        files.append(f)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archivo")
    ap.add_argument("--paginas", type=int, default=0, help="solo las N primeras (0 = todas)")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.archivo))[0]
    out_dir = os.path.join(ROOT, "output", name)
    os.makedirs(out_dir, exist_ok=True)
    pages = pages_of(args.archivo, out_dir, args.paginas)

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, use_safetensors=True,
                                      torch_dtype=torch.bfloat16).eval().cuda()

    result = os.path.join(out_dir, name + ".txt")
    with open(result, "w", encoding="utf-8") as fh:
        for i, page in enumerate(pages, 1):
            t0 = time.time()
            # el modelo imprime mucha depuración: se descarta
            with contextlib.redirect_stdout(io.StringIO()):
                text = model.infer(tok, image_file=page, output_path=out_dir, save_results=False)
            fh.write(f"\n===== Página {i} =====\n{text}\n")
            fh.flush()
            print(f"página {i}/{len(pages)} · {time.time() - t0:.0f}s", flush=True)
    print("guardado ->", result)


if __name__ == "__main__":
    main()

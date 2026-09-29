"""Local demo. The picture stays on this computer."""
import base64
import binascii
import json
import threading
import time
import urllib.parse
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from segment_denah import (
    analyze, outer_silhouette, analyze_organic, organic_silhouette, organic_traced_silhouette,
    rectilinear_hull_silhouette, instances_with_boxes, rectilinear_wall_fit_ratio,
    pdf_open, pdf_page_count, pdf_page_label, pdf_page_thumbnail_png,
    pdf_page_silhouette, pdf_page_rooms,
)

_ocr_engine = None


def _ocr():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    return _ocr_engine

ROOT = Path(__file__).resolve().parent

_CONTENT_TYPE_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _sanitize_stem(name, fallback="denah"):
    stem = Path(str(name)).stem
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:80]
    return stem or fallback


def _extension_for(content_type, filename):
    if filename:
        ext = Path(str(filename)).suffix.lower()
        if ext in _IMAGE_EXTS:
            return ext
    return _CONTENT_TYPE_EXT.get((content_type or "").lower(), ".png")


# ---------- background jobs: real progress, not a client-side animation. Detection runs
# in a background thread; the frontend polls /progress?id=<job_id> and shows whatever
# stage the backend actually reports -- if the pipeline hangs or crashes, the reported
# stage genuinely stops moving (or the job ends with an error) instead of a fake timer
# climbing regardless of what the backend is actually doing.
_jobs = {}
_jobs_lock = threading.Lock()
_JOB_TTL_SECONDS = 15 * 60  # finished jobs are pruned this long after completion


def _job_create():
    now = time.time()
    with _jobs_lock:
        stale = [jid for jid, j in _jobs.items() if j["done"] and j["finished_at"] and now - j["finished_at"] > _JOB_TTL_SECONDS]
        for jid in stale:
            del _jobs[jid]
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"stage": "Memulai", "step": 0, "total": 1, "done": False, "error": None, "result": None, "finished_at": None}
    return job_id


def _job_progress_cb(job_id):
    def cb(stage, step, total):
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["stage"] = stage
                job["step"] = step
                job["total"] = total
    return cb


def _job_finish(job_id, result=None, error=None):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["done"] = True
            job["result"] = result
            job["error"] = error
            job["finished_at"] = time.time()


def _job_get(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _rectilinear_rooms_look_broken(rooms, width, height):
    """Sanity check on analyze()'s rectilinear room boxes/polygons. When box_for()'s H/V
    wall search can't find a real wall in some direction (most sides, on a plan whose
    walls are mostly diagonal), it falls back to the whole silhouette bounding box for
    that side -- so several UNRELATED rooms end up sharing the exact same fallback
    rectangle. That's a much more specific tell than "this room looks big": a real
    multi-room plan doesn't have several different rooms with literally identical boxes,
    however large or small any single one of them is.
    """
    if not rooms:
        return False
    seen = {}
    for r in rooms:
        xs = [p[0] for p in r["points"]]
        ys = [p[1] for p in r["points"]]
        if not xs or not ys:
            continue
        key = (round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))
        seen[key] = seen.get(key, 0) + 1
    duplicated = sum(count for count in seen.values() if count > 1)
    return duplicated / len(rooms) > 0.3


def room_points(room):
    if room.get("points"):
        points = room["points"]
    else:
        left, top, right, bottom = room["box"]
        points = [[left, top], [right, top], [right, bottom], [left, bottom]]
    return {
        "id": room.get("id") or "",
        "name": room.get("name") or "",
        "source": room.get("source") or "kosong",
        "points": points,
    }
HOST = "127.0.0.1"
PORT = 8765


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/demo.html"):
            body = (ROOT / "demo.html").read_bytes()
            self._send(200, "text/html; charset=utf-8", body)
            return
        if self.path == "/contoh/Gambar3.jpeg":
            body = (ROOT / "contoh" / "Gambar3.jpeg").read_bytes()
            self._send(200, "image/jpeg", body)
            return
        if self.path.split("?")[0] == "/hasil":
            folder = ROOT / "hasil"
            names = sorted(p.name for p in folder.glob("*-siluet.json")) if folder.is_dir() else []
            self._send(200, "application/json", json.dumps(names).encode("utf-8"))
            return
        if self.path.startswith("/hasil/"):
            name = Path(self.path.split("?", 1)[0][len("/hasil/"):]).name
            target = ROOT / "hasil" / name
            if name.endswith("-siluet.json") and target.is_file():
                self._send(200, "application/json", target.read_bytes())
                return
        if self.path.split("?")[0] == "/cases":
            folder = ROOT / "hasil"
            names = sorted(p.name for p in folder.glob("*-case.json")) if folder.is_dir() else []
            self._send(200, "application/json", json.dumps(names).encode("utf-8"))
            return
        if self.path.startswith("/cases/"):
            name = Path(self.path.split("?", 1)[0][len("/cases/"):]).name
            target = ROOT / "hasil" / name
            if name.endswith("-case.json") and target.is_file():
                self._send(200, "application/json", target.read_bytes())
                return
        if self.path.split("?")[0] == "/progress":
            _, _, query = self.path.partition("?")
            job_id = urllib.parse.parse_qs(query).get("id", [""])[0]
            job = _job_get(job_id)
            if job is None:
                self._send(404, "application/json", b'{"error":"job_not_found"}')
                return
            job.pop("finished_at", None)
            self._send(200, "application/json", json.dumps(job).encode("utf-8"))
            return
        self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if self.path == "/save":
            self._save(raw)
            return
        if self.path == "/save_case":
            self._save_case(raw)
            return
        if self.path == "/delete_case":
            self._delete_case(raw)
            return
        path, _, query = self.path.partition("?")
        params = urllib.parse.parse_qs(query)
        if path == "/pdf_pages":
            self._pdf_pages(raw)
            return
        if path == "/pdf_render":
            self._pdf_render(raw, params)
            return
        if path == "/pdf_segment":
            self._pdf_segment(raw, params)
            return
        if path == "/segment_job":
            self._segment_job(raw, params)
            return
        if path == "/pdf_segment_job":
            self._pdf_segment_job(raw, params)
            return
        if path != "/segment":
            self.send_error(404)
            return
        mode = "siluet" if params.get("mode", [""])[0] == "siluet" else "ruang"
        # Two independent axes: `organik` picks the wall-DETECTION method (orientation-
        # agnostic thickness threshold, for curved/diagonal walls, vs the rectilinear H/V
        # line projection); `gabung` picks the silhouette STRATEGY (convex hull of the wall
        # material, for a mass split into several disconnected pieces meant as one site --
        # this can happen on a straight-walled plan too, not just a curved one -- vs a
        # single traced contour, which needs the wall material to form one connected mass).
        organik = params.get("organik", ["0"])[0] in ("1", "true")
        gabung = params.get("gabung", ["0"])[0] in ("1", "true")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self._send(400, "application/json", b'{"error":"gambar tidak terbaca"}')
            return
        height, width = image.shape[:2]
        adjacency = []
        if mode == "siluet":
            if organik and gabung:
                words, _ = _ocr()(image)
                points = organic_silhouette(image, boxed_labels=instances_with_boxes(words))
            elif organik:
                words, _ = _ocr()(image)
                points = organic_traced_silhouette(image, boxed_labels=instances_with_boxes(words))
            elif gabung:
                points = rectilinear_hull_silhouette(image)
            else:
                points = outer_silhouette(image)
            rooms = [{"name": "", "source": "kosong", "points": points}] if points else []
        elif organik:
            # Room-boundary tracing for walls with door gaps isn't reliably solved yet (see
            # segment_denah.analyze_organic docstring) -- each detected label comes back as
            # a nearest-wall rectangle (ray-cast against the wall mask, same idea as the
            # rectilinear pipeline's nearest-wall fallback box but working off the raw
            # mask instead of pre-extracted line segments) approximating its real floor
            # area, not a fabricated exact shape, meant to be reshaped by hand in the
            # existing polygon editor.
            _vis, _sil, labels_out = analyze_organic(image)
            rooms = [{"name": lab["name"], "source": lab.get("source", "kotak-organik"), "points": lab["points"]} for lab in labels_out]
        else:
            _vis, found, adjacency = analyze(image)
            rooms = [room_points(r) for r in found]
        self._send(200, "application/json", json.dumps({
            "width": width,
            "height": height,
            "mode": mode,
            "rooms": rooms,
            "adjacency": adjacency,
        }).encode("utf-8"))

    def _pdf_pages(self, raw):
        """Page count + a small preview thumbnail per page, for the frontend's page
        picker -- shown right after a PDF is uploaded through the single unified input,
        before the user chooses which page (denah lantai) to analyze.
        """
        try:
            count = pdf_page_count(raw)
        except RuntimeError:
            self._send(500, "application/json", b'{"error":"pymupdf_missing"}')
            return
        doc = pdf_open(raw)
        pages = []
        for i in range(count):
            thumb = base64.b64encode(pdf_page_thumbnail_png(raw, i)).decode("ascii")
            pages.append({
                "index": i,
                "label": pdf_page_label(doc, i),
                "thumbnail": "data:image/png;base64," + thumb,
            })
        self._send(200, "application/json", json.dumps({"pages": pages}).encode("utf-8"))

    def _pdf_render(self, raw, params):
        """Just the rendered page as a PNG, at the SAME zoom /pdf_segment uses for its
        analysis coordinate space -- lets the frontend show a background image right
        after the user picks a page, without waiting on silhouette/room detection.
        """
        try:
            page_index = int(params.get("page", ["0"])[0])
        except ValueError:
            page_index = 0
        zoom = 2.0
        try:
            png_bytes = pdf_page_thumbnail_png(raw, page_index, zoom=zoom)
        except RuntimeError:
            self._send(500, "application/json", b'{"error":"pymupdf_missing"}')
            return
        except IndexError:
            self._send(400, "application/json", b'{"error":"halaman tidak ditemukan"}')
            return
        arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
        height, width = arr.shape[:2]
        image_base64 = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        self._send(200, "application/json", json.dumps({
            "width": width,
            "height": height,
            "image_base64": image_base64,
        }).encode("utf-8"))

    def _pdf_segment(self, raw, params):
        """Same response shape as /segment (width, height, mode, rooms, adjacency), plus
        an `image_base64` data URL of the page rendered at the SAME zoom used to compute
        every point/box, so the frontend can drop it in as the panel's background image
        with coordinates that line up exactly -- no separate raster upload needed for PDFs.
        """
        try:
            page_index = int(params.get("page", ["0"])[0])
        except ValueError:
            page_index = 0
        mode = "siluet" if params.get("mode", [""])[0] == "siluet" else "ruang"
        zoom = 2.0
        try:
            if mode == "siluet":
                bgr, points = pdf_page_silhouette(raw, page_index, zoom=zoom)
                rooms = [{"name": "", "source": "kosong", "points": points}] if points else []
                adjacency = []
            else:
                bgr, _sil, found = pdf_page_rooms(raw, page_index, zoom=zoom)
                rooms = [{"name": r["name"], "source": "kotak-pdf", "points": r["points"]} for r in found]
                adjacency = []
        except RuntimeError:
            self._send(500, "application/json", b'{"error":"pymupdf_missing"}')
            return
        except IndexError:
            self._send(400, "application/json", b'{"error":"halaman tidak ditemukan"}')
            return
        height, width = bgr.shape[:2]
        ok, buf = cv2.imencode(".png", bgr)
        image_base64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        self._send(200, "application/json", json.dumps({
            "width": width,
            "height": height,
            "mode": mode,
            "rooms": rooms,
            "adjacency": adjacency,
            "image_base64": image_base64,
        }).encode("utf-8"))

    def _segment_job(self, raw, params):
        """Async counterpart of /segment: decodes and validates the image synchronously
        (fast), then runs the actual detection in a background thread reporting real
        progress, and returns a job_id right away for the frontend to poll at /progress.
        """
        mode = "siluet" if params.get("mode", [""])[0] == "siluet" else "ruang"
        organik = params.get("organik", ["0"])[0] in ("1", "true")
        gabung = params.get("gabung", ["0"])[0] in ("1", "true")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            self._send(400, "application/json", b'{"error":"gambar tidak terbaca"}')
            return
        height, width = image.shape[:2]
        job_id = _job_create()
        progress = _job_progress_cb(job_id)

        def work():
            try:
                adjacency = []
                if mode == "siluet":
                    if organik and gabung:
                        words, _ = _ocr()(image)
                        points = organic_silhouette(image, boxed_labels=instances_with_boxes(words))
                    elif organik:
                        words, _ = _ocr()(image)
                        points = organic_traced_silhouette(image, boxed_labels=instances_with_boxes(words))
                    elif gabung:
                        points = rectilinear_hull_silhouette(image)
                    else:
                        points = outer_silhouette(image, progress=progress)
                    rooms = [{"name": "", "source": "kosong", "points": points}] if points else []
                elif organik:
                    _vis, _sil, labels_out = analyze_organic(image, progress=progress)
                    rooms = [{"name": lab["name"], "source": lab.get("source", "kotak-organik"), "points": lab["points"]} for lab in labels_out]
                elif rectilinear_wall_fit_ratio(image) < 0.7:
                    # Checked BEFORE running the H/V pipeline at all: if most of this
                    # drawing's real wall material isn't axis-aligned (a circular/radial
                    # plan, a hexagon, ...), analyze()'s room tracer is doomed regardless
                    # of how many OCR labels happen to be readable -- verified this
                    # direct geometric signal is far more reliable than checking analyze()'s
                    # OWN output afterward for symptoms like duplicate boxes (see
                    # _rectilinear_rooms_look_broken below, kept as a secondary net for
                    # whatever this upfront check doesn't catch): a genuinely good
                    # rectilinear result can have just as many "leftover circulation blob"
                    # or "duplicate box" quirks as a genuinely broken circular one, so
                    # symptom-counting after the fact doesn't generalize as well as
                    # measuring the wall geometry itself up front.
                    _vis, _sil, labels_out = analyze_organic(image, progress=progress)
                    rooms = [{"name": lab["name"], "source": lab.get("source", "kotak-organik"), "points": lab["points"]} for lab in labels_out]
                else:
                    _vis, found, adjacency = analyze(image, progress=progress)
                    rooms = [room_points(r) for r in found]
                    if _rectilinear_rooms_look_broken(rooms, width, height):
                        # Secondary safety net for whatever the upfront wall-geometry
                        # check above didn't catch: the H/V line-projection room tracer
                        # needs walls it can capture as horizontal/vertical runs, so
                        # box_for() falls back to the whole silhouette bounding box for
                        # missing sides and rooms come back sharing implausible duplicate
                        # boxes. Re-run with the orientation-agnostic ray-box/tracing
                        # method instead (same one used for curved/circular plans).
                        _vis2, _sil2, labels_out = analyze_organic(image, progress=progress)
                        rooms = [{"name": lab["name"], "source": lab.get("source", "kotak-organik"), "points": lab["points"]} for lab in labels_out]
                        adjacency = []
                _job_finish(job_id, result={
                    "width": width, "height": height, "mode": mode, "rooms": rooms, "adjacency": adjacency,
                })
            except Exception as exc:
                _job_finish(job_id, error=str(exc))

        threading.Thread(target=work, daemon=True).start()
        self._send(200, "application/json", json.dumps({"job_id": job_id}).encode("utf-8"))

    def _pdf_segment_job(self, raw, params):
        """Async counterpart of /pdf_segment, same job/poll pattern as _segment_job."""
        try:
            page_index = int(params.get("page", ["0"])[0])
        except ValueError:
            page_index = 0
        mode = "siluet" if params.get("mode", [""])[0] == "siluet" else "ruang"
        zoom = 2.0
        job_id = _job_create()
        progress = _job_progress_cb(job_id)

        def work():
            try:
                if mode == "siluet":
                    bgr, points = pdf_page_silhouette(raw, page_index, zoom=zoom, progress=progress)
                    rooms = [{"name": "", "source": "kosong", "points": points}] if points else []
                    adjacency = []
                else:
                    bgr, _sil, found = pdf_page_rooms(raw, page_index, zoom=zoom, progress=progress)
                    rooms = [{"name": r["name"], "source": "kotak-pdf", "points": r["points"]} for r in found]
                    adjacency = []
                height, width = bgr.shape[:2]
                ok, buf = cv2.imencode(".png", bgr)
                image_base64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
                _job_finish(job_id, result={
                    "width": width, "height": height, "mode": mode,
                    "rooms": rooms, "adjacency": adjacency, "image_base64": image_base64,
                })
            except RuntimeError:
                _job_finish(job_id, error="pymupdf_missing")
            except IndexError:
                _job_finish(job_id, error="halaman tidak ditemukan")
            except Exception as exc:
                _job_finish(job_id, error=str(exc))

        threading.Thread(target=work, daemon=True).start()
        self._send(200, "application/json", json.dumps({"job_id": job_id}).encode("utf-8"))

    def _save(self, raw):
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, "application/json", b'{"error":"json tidak terbaca"}')
            return
        mode = data.get("mode")
        if mode not in ("ruang", "siluet"):
            self._send(400, "application/json", b'{"error":"mode tidak dikenal"}')
            return
        stem = _sanitize_stem(data.get("image") or "denah")
        folder = ROOT / "hasil"
        folder.mkdir(exist_ok=True)
        target = folder / f"{stem}-{mode}.json"
        record = {
            "image": data.get("image") or "",
            "mode": mode,
            "polygons": data.get("polygons") or [],
        }
        target.write_text(json.dumps(record, indent=2), encoding="utf-8")
        self._send(200, "application/json", json.dumps({"saved": target.name}).encode("utf-8"))

    def _save_case(self, raw):
        """One self-contained case per denah: the image, its silhouette polygon, room
        polygons, and door adjacency -- everything the next similarity components (beyond
        silhouette) will need to read back later, in one place instead of scattered across
        separate -siluet/-ruang files. The image is saved BOTH ways: as a real file next to
        the JSON (so it can be opened/browsed directly, or read by other scripts with plain
        cv2.imread), and embedded as base64 inside the JSON (so the JSON alone still stands
        on its own, e.g. if copied elsewhere without its sibling image file).
        """
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, "application/json", b'{"error":"json tidak terbaca"}')
            return
        image_meta = data.get("image") or {}
        stem = _sanitize_stem(data.get("name") or image_meta.get("filename") or "denah")
        folder = ROOT / "hasil"
        folder.mkdir(exist_ok=True)

        image_file = None
        if image_meta.get("data_base64"):
            try:
                image_bytes = base64.b64decode(image_meta["data_base64"])
            except (binascii.Error, ValueError):
                image_bytes = None
            if image_bytes:
                ext = _extension_for(image_meta.get("content_type"), image_meta.get("filename"))
                image_path = folder / f"{stem}{ext}"
                image_path.write_bytes(image_bytes)
                image_file = image_path.name

        target = folder / f"{stem}-case.json"
        record = {
            "name": stem,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "image": data.get("image") or None,
            "image_file": image_file,
            "width": data.get("width"),
            "height": data.get("height"),
            "silhouette": data.get("silhouette"),
            "rooms": data.get("rooms") or [],
            "adjacency": data.get("adjacency") or [],
            # Peta Aksial (Component 3 NAIN/NACH input) -- was missing from this explicit
            # field list entirely, so it silently never made it into the saved case JSON
            # even though the frontend always sent it (per-room fields like
            # manualExterior survived fine, riding along inside "rooms" above; this is a
            # separate top-level field and needed its own entry here).
            "axialLines": data.get("axialLines") or [],
        }
        target.write_text(json.dumps(record, indent=2), encoding="utf-8")
        self._send(200, "application/json", json.dumps({
            "saved": target.name,
            "image_saved": image_file,
        }).encode("utf-8"))

    def _delete_case(self, raw):
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, "application/json", b'{"error":"json tidak terbaca"}')
            return
        name = Path(str(data.get("name") or "")).name
        if not name.endswith("-case.json"):
            self._send(400, "application/json", b'{"error":"nama kasus tidak valid"}')
            return
        folder = ROOT / "hasil"
        target = folder / name
        if not target.is_file():
            self._send(404, "application/json", b'{"error":"kasus tidak ditemukan"}')
            return
        image_file = None
        try:
            record = json.loads(target.read_text(encoding="utf-8"))
            image_file = record.get("image_file")
        except (json.JSONDecodeError, OSError):
            pass
        target.unlink()
        if image_file:
            image_path = folder / Path(image_file).name
            if image_path.is_file():
                try:
                    image_path.unlink()
                except OSError:
                    pass
        self._send(200, "application/json", json.dumps({"deleted": name}).encode("utf-8"))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(fmt % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Demo siap di http://{HOST}:{PORT}/")
    server.serve_forever()

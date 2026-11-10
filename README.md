# gpt-sovits-server

Slim, GPU-only **GPT-SoVITS v2** inference server (`server_fast.py`), packaged for
the k3s cluster and served **only via ClusterIP** (no Ingress).

## What this image contains

| Included | Excluded (on purpose) |
|---|---|
| `server_fast.py` (OpenAI-compatible `/v1/audio/speech`) | training code |
| `GPT_SoVITS` inference modules | ASR / UVR5 / denoiser |
| `tools/` runtime helpers (`audio_sr`, `acceleration`, `i18n`) | Gradio WebUI |
| `voices/` reference audio | model weights (mounted) |

Model weights are **not** baked in. They are mounted from the host at runtime:

```
/app/GPT_SoVITS/pretrained_models   -> /home/k3s/data/gpt-sovits/pretrained_models
/app/GPT_SoVITS/text/G2PWModel      -> /home/k3s/data/gpt-sovits/G2PWModel
/app/voices                         -> /home/k3s/data/gpt-sovits/voices
```

### Required model layout (v2, ~1.9G)

```
/home/k3s/data/gpt-sovits/
├── pretrained_models/
│   ├── chinese-roberta-wwm-ext-large/      (~622M)
│   ├── chinese-hubert-base/                (~189M)
│   ├── gsv-v2final-pretrained/             (~339M)
│   │   ├── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
│   │   └── s2G2333k.pth
│   └── fast_langdetect/
│       └── lid.176.bin                     (~126M)
├── G2PWModel/                              (~608M)
└── voices/
    ├── female_nurse.wav
    └── female_nurse.txt
```

## Build

Automatically built by GitHub Actions on push to `main`:

```text
ghcr.io/sdlizhan/gpt-sovits-server:v2.4
```

Push builds publish `edge`; release tags are created explicitly (see
`.github/workflows/build.yml` and `.github/workflows/hotfix.yml`).

Manual build:

```bash
docker build -t ghcr.io/sdlizhan/gpt-sovits-server:v2.4 .
```

## Deploy

```bash
kubectl apply -f k8s/tts.yml
kubectl -n tts get pods -o wide
kubectl -n tts logs deploy/gpt-sovits-server -f
```

In-cluster endpoint:

```text
http://gpt-sovits-service.tts.svc.cluster.local:50045
```

There is intentionally **no Ingress**. From outside the cluster use
`kubectl -n tts port-forward svc/gpt-sovits-service 50045:50045`.

## API

```bash
curl -s http://gpt-sovits-service.tts.svc.cluster.local:50045/health
curl -s http://gpt-sovits-service.tts.svc.cluster.local:50045/v1/audio/voices
curl -s -X POST http://gpt-sovits-service.tts.svc.cluster.local:50045/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"该吃药了","voice":"female_nurse","response_format":"wav"}' \
  --output out.wav
```

`language` defaults to `zh`. `en` is also supported. `ja/ko/yue` are not
installed in this slim image and will fail.

## Notes

- `imagePullPolicy: IfNotPresent` with a fixed tag: restarts use the cached
  image and never depend on registry DNS.
- `strategy: Recreate` avoids two pods competing for the single GPU.
- GPU prerequisites: PCIe passthrough to the k3s VM, `nvidia-open`,
  `nvidia-container-toolkit`, RuntimeClass `nvidia`. No NVIDIA device plugin
  is used: the pod runs under the `nvidia` runtime with all GPUs visible.
- `speed` is applied inside the model sampler, so no `ffmpeg` binary is needed.

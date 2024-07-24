### NiiVue ONNX

ONNX 3D convolution test for [brainchop](https://github.com/neuroneural/brainchop) models.

### For Developers

You can serve a hot-reloadable web page that allows you to interactively modify the source code.

```bash
git clone git@github.com:niivue/niivue-onnx.git
cd niivue-onnx
npm install
npm run dev # Only works in Chrome due to WebGPU usage
```

#### to build and serve the built version

```bash
npm run build
npx http-server dist/ # Only works in Chrome due to WebGPU usage
```


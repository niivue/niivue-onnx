import { Niivue } from '@niivue/niivue'
// IMPORTANT: we need to import this specific file. 
import * as ort from "./node_modules/onnxruntime-web/dist/ort.all.mjs"
console.log(ort);
async function main() {
  aboutBtn.onclick = function () {
    let url = "https://github.com/axinging/mlmodel-convension-demo/blob/main/onnx/onnx-brainchop.html"
    window.open(url, '_blank').focus();
  }
  function handleLocationChange(data) {
    document.getElementById("intensity").innerHTML = data.string
  }
  const defaults = {
    backColor: [0.4, 0.4, 0.4, 1],
    show3Dcrosshair: true,
    onLocationChange: handleLocationChange,
    dragAndDropEnabled: false,
  }
  const nv1 = new Niivue(defaults)
  nv1.attachToCanvas(gl1)
  await nv1.loadVolumes([{ url: './t1_crop.nii.gz' }])
  // FIXME: Do we want to conform?
  /*const conformed = await nv1.conform(
    nv1.volumes[0],
    false,
    true,
    true
  )
  nv1.removeVolume(nv1.volumes[0])
  nv1.addVolume(conformed)*/

  let img32 = new Float32Array(nv1.volumes[0].img)
  let mx = img32[0]
  let mn = mx
  for (let i = 0; i < img32.length; i++) {
    mx = Math.max(mx, img32[i])
    mn = Math.min(mn, img32[i])
  }
  let scale32 = 1 / (mx - mn)
  for (let i = 0; i < img32.length; i++) {
    img32[i] = (img32[i] - mn) * scale32
  }
  let feedsInfo = [];
  function getFeedInfo(feed, type, data, dims) {
    const warmupTimes = 0;
    const runTimes = 1;
    for (let i = 0; i < warmupTimes + runTimes; i++) {
      let typedArray;
      let typeBytes;
      if (type === 'bool') {
        data = [data];
        dims = [1];
        typeBytes = 1;
      } else if (type === 'int8') {
        typedArray = Int8Array;
      } else if (type === 'float16') {
        typedArray = Uint16Array;
      } else if (type === 'int32') {
        typedArray = Int32Array;
      } else if (type === 'uint32') {
        typedArray = Uint32Array;
      } else if (type === 'float32') {
        typedArray = Float32Array;
      } else if (type === 'int64') {
        typedArray = BigInt64Array;
      }
      if (typeBytes === undefined) {
        typeBytes = typedArray.BYTES_PER_ELEMENT;
      }

      let size, _data;
      if (Array.isArray(data) || ArrayBuffer.isView(data)) {
        size = data.length;
        _data = data;
      } else {
        size = dims.reduce((a, b) => a * b);
        if (data === 'random') {
          _data = typedArray.from({ length: size }, () => getRandom(type));
        } else {
          _data = typedArray.from({ length: size }, () => data);
        }
      }

      if (i > feedsInfo.length - 1) {
        feedsInfo.push(new Map());
      }
      feedsInfo[i].set(feed, [type, _data, dims, Math.ceil(size * typeBytes / 16) * 16]);
    }
    return feedsInfo;
  }
  const option = {
    executionProviders: [
      {
        name: 'webgpu',
      },
    ],
    graphOptimizationLevel: 'extended',
    optimizedModelFilepath: 'opt.onnx'
  };

  const session = await ort.InferenceSession.create('./model_5_channels.onnx', option);
  const shape = [1, 1, 256, 256, 256];
  // FIXME: Do we want to use a real image for inference?
  const imgData = img32;
  const expectedLength = shape.reduce((a, b) => a * b);
  // FIXME: Do we need want this?
  if (imgData.length !== expectedLength) {
    throw new Error(`imgData length (${imgData.length}) does not match expected tensor length (${expectedLength})`);
  }

  const temp = getFeedInfo("input.1", "float32", imgData, shape);
  let dataA = temp[0].get('input.1')[1];
  const tensorA = new ort.Tensor('float32', dataA, shape);

  const feeds = { "input.1": tensorA };
  // feed inputs and run
  const results = await session.run(feeds);
  console.log(results);
  const aiVox = results[39].data
  const outDims = results[39].dims
  const vols = outDims[1]
  const vox = outDims[2] * outDims[3] * outDims[4]
  if ((img32.length != vox) || (vols != 3) || (aiVox.length != (vols * vox))) {
    console.log('Fatal error')
  }
  const outData = new Float32Array(vox)
  for (let i = 0; i < vox; i++) {
    outData[i] = Math.max(Math.max(aiVox[i],aiVox[i+vox]),aiVox[i+vox+vox])
  }
  const newImg = nv1.cloneVolume(0);
  newImg.img = outData
  newImg.cal_min = 3
  newImg.cal_max = 4
  newImg.hdr.datatypeCode = 16
  newImg.hdr.dims[4] = 1
  newImg.trustCalMinMax = false
  console.log(newImg)
  // Add the output to niivue
  nv1.addVolume(newImg)
  nv1.setColormap(newImg.id, "actc")
  nv1.setOpacity(1, 0.5)
}

main()

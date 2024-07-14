import { Niivue } from '@niivue/niivue'
import * as ort from 'onnxruntime-web';
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
              //name: 'webgpu',
              name: 'webgl',
          },
      ],
      graphOptimizationLevel: 'extended',
      optimizedModelFilepath: 'opt.onnx'
  };

  const session = await ort.InferenceSession.create('./model_5_channels.onnx', option);
  const shape = [1, 1, 256, 256, 256];
  const temp = getFeedInfo("input.1", "float32", 0, shape);
  let dataA = temp[0].get('input.1')[1];
  // let dataTemp = await loadJSON("./onnx-branchchop-input64.jsonc");
  // dataA = dataTemp['data'];
  const tensorA = new ort.Tensor('float32', dataA, shape);
  
  const feeds = { "input.1": tensorA };
  // feed inputs and run
  console.log("before run");
  const results = await session.run(feeds);
  console.log("after run");
}

main()

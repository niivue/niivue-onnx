import { Niivue } from '@niivue/niivue'
// IMPORTANT: we need to import this specific file. 
import * as ort from "./node_modules/onnxruntime-web/dist/ort.all.mjs"
async function main() {
  async function ensureConformed() {
    const nii = nv1.volumes[0]
    let isConformed = nii.dims[1] === 256 && nii.dims[2] === 256 && nii.dims[3] === 256
    if (nii.permRAS[0] !== -1 || nii.permRAS[1] !== 3 || nii.permRAS[2] !== -2) {
      isConformed = false
    }
    if (isConformed) {
      return
    }
    const nii2 = await nv1.conform(nii, false)
    await nv1.removeVolume(nv1.volumes[0])
    await nv1.addVolume(nii2)
  }
  async function closeAllOverlays() {
    while (nv1.volumes.length > 1) {
      await nv1.removeVolume(nv1.volumes[1])
    }
  }
  segmentBtn.onclick = async function () {
    if (nv1.volumes.length < 1) {
      window.alert('Please open a voxel-based image')
      return
    }
    await closeAllOverlays()
    await ensureConformed()
    let img32 = new Float32Array(nv1.volumes[0].img)
    // normalize input data to range 0..1
    //  TODO: ONNX not JavaScript https://onnx.ai/onnx/operators/onnx_aionnxml_Normalizer.html
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
    // load onnx model
    const option = {
      executionProviders: [
        {
          name: 'webgpu',
        },
      ],
      graphOptimizationLevel: 'extended',
      optimizedModelFilepath: 'opt.onnx'
    }
    const session = await ort.InferenceSession.create('./model.onnx', option)
    const shape = [1, 1, 256, 256, 256]
    const nvox = shape.reduce((a, b) => a * b)
    if (img32.length !== nvox) {
      throw new Error(`img32 length (${img32.length}) does not match expected tensor length (${expectedLength})`)
    }
    const imgTensor = new ort.Tensor('float32', img32, shape)
    const feeds = { "input": imgTensor }
    // run onnx inference
    const results = await session.run(feeds)
    const classImg = results.output.cpuData
    // classImg will have one volume per class
    const nvol = Math.floor(classImg.length / nvox)
    if ((nvol < 2) || (classImg.length != (nvol * nvox))) {
      console.log('Fatal error')
    }
    console.log(`${nvol} volumes each with ${nvox} voxels`)
    // argmax should identify correct class for each voxel
    //  TODO: ONNX not JavaScript https://onnx.ai/onnx/operators/onnx__ArgMax.html
    const argMaxImg = new Float32Array(nvox)
    for (let vox = 0; vox < nvox; vox++) {
      let mxVal = classImg[vox]
      let mxVol = 0
      for (let vol = 1; vol < nvol; vol++) {
        const val = classImg[vox + (vol * nvox)]
        if (val > mxVal) {
          mxVol = vol
          mxVal = val
        }
      }
      // Next line incorrect!
      argMaxImg[vox] = mxVal
      // Next line should be correct: brightest volume
      // argMaxImg[vox] = mxVol
    }
    //
    const newImg = nv1.cloneVolume(0)
    newImg.img = argMaxImg
    newImg.hdr.datatypeCode = 16 // = float32
    newImg.hdr.dims[4] = 1
    newImg.trustCalMinMax = false
    console.log(newImg)
    // Add the output to niivue
    nv1.addVolume(newImg)
    nv1.setColormap(newImg.id, "actc")
    nv1.setOpacity(1, 0.5)
  }
  function handleLocationChange(data) {
    document.getElementById("intensity").innerHTML = data.string
  }
  const defaults = {
    backColor: [0.4, 0.4, 0.4, 1],
    onLocationChange: handleLocationChange,
  }
  const nv1 = new Niivue(defaults)
  nv1.attachToCanvas(gl1)
  await nv1.loadVolumes([{ url: './t1_crop.nii.gz' }])
  segmentBtn.onclick()
}

main()

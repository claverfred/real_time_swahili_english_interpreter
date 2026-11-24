// Collects microphone samples (first channel) into ~100 ms Float32 blocks for the page.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = Math.round(sampleRate / 10);
    this.buf = new Float32Array(this.block);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      let i = 0;
      while (i < ch.length) {
        const take = Math.min(ch.length - i, this.block - this.n);
        this.buf.set(ch.subarray(i, i + take), this.n);
        this.n += take; i += take;
        if (this.n === this.block) {
          this.port.postMessage(this.buf, [this.buf.buffer]);
          this.buf = new Float32Array(this.block);
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);

import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';

export const TURBO_DEFAULTS = {
  fl4: 'h3_turbo/minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors',
  ref4: 'h3_turbo/minimax_h3_ref2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors',
  fl8: 'h3_turbo/minimax_h3_fl2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors',
  ref8: 'h3_turbo/minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors',
};
export const defaultConfig = () => ({
  comfy: { url: 'http://127.0.0.1:8188', token: '' },
  llm: { url: 'http://127.0.0.1:11434/v1', model: '', key: '' },
  models: { videoModel: '', referenceModel: '', textEncoder: '', videoVae: '', audioVae: '', imageModel: '', imageTextEncoder: '', imageVae: '', turboSteps: 0, turboFl4: TURBO_DEFAULTS.fl4, turboRef4: TURBO_DEFAULTS.ref4, turboFl8: TURBO_DEFAULTS.fl8, turboRef8: TURBO_DEFAULTS.ref8 },
});

export class Store {
  constructor(directory) { this.directory = directory; this.pending = Promise.resolve(); }
  async init() {
    await mkdir(path.join(this.directory, 'media'), { recursive: true });
    try { this.data = JSON.parse(await readFile(path.join(this.directory, 'studio.json'), 'utf8')); }
    catch (error) {
      if (error.code !== 'ENOENT') throw error;
      this.data = { config: defaultConfig(), messages: [], assets: [], jobs: [], createdAt: new Date().toISOString() };
    }
    this.data.busy = false;
    this.data.messages = this.data.messages.filter(m => !m.pending);
    return this;
  }
  id() { return randomUUID(); }
  message(role, content, extra = {}) {
    const message = { id: this.id(), role, content, createdAt: new Date().toISOString(), ...extra };
    this.data.messages.push(message);
    return message;
  }
  save() {
    const snapshot = JSON.stringify(this.data, null, 2);
    const operation = this.pending.then(async () => {
      const temporary = path.join(this.directory, 'studio.json.tmp');
      await writeFile(temporary, snapshot, { mode: 0o600 });
      await rename(temporary, path.join(this.directory, 'studio.json'));
    });
    this.pending = operation.catch(() => {});
    return operation;
  }
  publicState() {
    const { config, ...state } = this.data;
    return { ...state, assets: state.assets.filter(asset => !asset.deletedAt), jobs: state.jobs.filter(job => !job.deletedAt).map(({ connection, ...job }) => job), config: {
      ...config,
      comfy: { url: config.comfy.url, hasToken: Boolean(config.comfy.token) },
      llm: { url: config.llm.url, model: config.llm.model, hasKey: Boolean(config.llm.key) },
    } };
  }
}

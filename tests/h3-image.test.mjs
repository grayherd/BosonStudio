import test from 'node:test';
import assert from 'node:assert/strict';
import { validateGeneration, buildWorkflow } from '../lib/workflows.mjs';

const models = { videoModel: 'fl.safetensors', referenceModel: 'ref.safetensors', textEncoder: 'clip.safetensors', videoVae: 'vvae.safetensors', audioVae: 'avae.safetensors', imageModel: 'minimax_h3_image_pruned_bf16.safetensors', imageTextEncoder: 'h3/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors', imageVae: 'minimax_h3_image_vae_fp16.safetensors' };
const nodesOf = (graph, type) => Object.values(graph).filter(n => n.class_type === type);

test('text_to_image uses the native H3-Image graph with 6 steps', () => {
  const { graph, metadata } = buildWorkflow(validateGeneration({ mode: 'text_to_image', prompt: 'a blue ceramic cup on a wooden table', aspect_ratio: 'square', turbo: 4 }, []), models, new Map());
  assert.equal(nodesOf(graph, 'MiniMaxH3ImageConditioning').length, 1);
  assert.equal(nodesOf(graph, 'KSampler')[0].inputs.steps, 6);
  assert.equal(nodesOf(graph, 'BasicScheduler').length, 0);
  assert.equal(nodesOf(graph, 'LoraLoaderModelOnly').length, 0);
  assert.deepEqual([metadata.width, metadata.height], [4096, 4096]);
});

test('config overrides for image model files are honored', () => {
  const { graph } = buildWorkflow(validateGeneration({ mode: 'text_to_image', prompt: 'an editorial photo of an orange chair', aspect_ratio: 'landscape' }, []), { ...models, imageModel: 'custom_image.safetensors', imageVae: 'custom_vae.safetensors' }, new Map());
  assert.equal(nodesOf(graph, 'UNETLoader')[0].inputs.unet_name, 'custom_image.safetensors');
  assert.equal(nodesOf(graph, 'VAELoader')[0].inputs.vae_name, 'custom_vae.safetensors');
});

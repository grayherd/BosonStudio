import test from 'node:test';
import assert from 'node:assert/strict';
import { validateGeneration, buildWorkflow, TURBO_LORAS } from '../lib/workflows.mjs';
import { parseXmlToolCalls } from '../lib/assistant.mjs';

const models = { videoModel: 'fl.safetensors', referenceModel: 'ref.safetensors', textEncoder: 'clip.safetensors', videoVae: 'vvae.safetensors', audioVae: 'avae.safetensors' };
const nodesOf = (graph, type) => Object.values(graph).filter(n => n.class_type === type);

test('full run: 20 steps, no LoRA node', () => {
  const { graph } = buildWorkflow(validateGeneration({ mode: 'text_to_video', prompt: 'a paper boat on a stream', duration: 5, aspect_ratio: 'landscape' }, []), models, new Map());
  assert.equal(nodesOf(graph, 'LoraLoaderModelOnly').length, 0);
  assert.equal(nodesOf(graph, 'BasicScheduler')[0].inputs.steps, 20);
});

test('turbo 4 inserts fl2v LoRA and 4 steps', () => {
  const { graph } = buildWorkflow(validateGeneration({ mode: 'text_to_video', prompt: 'a paper boat on a stream', duration: 5, aspect_ratio: 'landscape', turbo: 4 }, []), models, new Map());
  const [lora] = nodesOf(graph, 'LoraLoaderModelOnly');
  assert.equal(lora.inputs.lora_name, TURBO_LORAS.fl[4]);
  assert.equal(nodesOf(graph, 'BasicScheduler')[0].inputs.steps, 4);
});

test('turbo 8 on reference mode picks ref2v LoRA', () => {
  const asset = { id: 'a1', kind: 'image', metadata: { width: 1024, height: 1024 } };
  const { graph } = buildWorkflow(validateGeneration({ mode: 'reference_to_video', prompt: 'animate the character', duration: 5, asset_ids: ['a1'], turbo: 8 }, [asset]), models, new Map([['a1', 'a1.png']]));
  const [lora] = nodesOf(graph, 'LoraLoaderModelOnly');
  assert.equal(lora.inputs.lora_name, TURBO_LORAS.ref[8]);
  assert.equal(nodesOf(graph, 'BasicScheduler')[0].inputs.steps, 8);
});

test('config-provided LoRA name overrides the built-in default', () => {
  const { graph } = buildWorkflow(validateGeneration({ mode: 'text_to_video', prompt: 'a paper boat on a stream', duration: 5, aspect_ratio: 'landscape', turbo: 4 }, []), { ...models, turboFl4: 'h3_turbo/custom_future_lora.safetensors' }, new Map());
  const [lora] = nodesOf(graph, 'LoraLoaderModelOnly');
  assert.equal(lora.inputs.lora_name, 'h3_turbo/custom_future_lora.safetensors');
});

test('still-image modes ignore turbo (20 steps, no LoRA)', () => {
  const { graph } = buildWorkflow(validateGeneration({ mode: 'text_to_image', prompt: 'an older man fishing from a dock', aspect_ratio: 'landscape', turbo: 4 }, []), models, new Map());
  assert.equal(nodesOf(graph, 'LoraLoaderModelOnly').length, 0);
  assert.equal(nodesOf(graph, 'KSampler')[0].inputs.steps, 6);
});

test('XML tool-call fallback parses Qwen content blocks', () => {
  const P = 'parameter';
  const xml = '<tool_call><function=get_job_status><' + P + '=job_id>abc-123</' + P + '></function></tool_call>';
  const calls = parseXmlToolCalls(xml);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].function.name, 'get_job_status');
  assert.deepEqual(JSON.parse(calls[0].function.arguments), { job_id: 'abc-123' });
});

test('image modes ignore turbo and keep the full recipe', () => {
  const { graph } = buildWorkflow(validateGeneration({ mode: 'text_to_image', prompt: 'a blue ceramic cup', aspect_ratio: 'square', turbo: 4 }, []), models, new Map());
  assert.equal(nodesOf(graph, 'LoraLoaderModelOnly').length, 0);
  assert.equal(nodesOf(graph, 'KSampler')[0].inputs.steps, 6);
});

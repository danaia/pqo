<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { AFFECT_COLORS, MODE_COLORS, POPULATION, initializeAgents, type ColorMode, type Layout } from './simulation'

const props = defineProps<{
  layout: Layout
  colorMode: ColorMode
  paused: boolean
  minute: number
  agency: number
  transit: number
  population: number
  scenario: string
  zoom: number
  resetToken: number
}>()
const emit = defineEmits<{ select: [id: number] }>()
const canvas = ref<HTMLCanvasElement | null>(null)
const field = initializeAgents()
let animation = 0
let context: CanvasRenderingContext2D | null = null
let width = 0
let height = 0
let dpr = 1
let simulationTime = 0
let lastFrame = performance.now()

const layoutRotation: Record<Layout, number> = { manhattan: -0.62, barcelona: -0.45, tokyo: -0.73 }

type Point = readonly [number, number]

function project(x: number, y: number, z = 0): Point {
  const rotation = layoutRotation[props.layout]
  const cos = Math.cos(rotation)
  const sin = Math.sin(rotation)
  const rx = x * cos - y * sin
  const ry = x * sin + y * cos
  const scale = Math.min(width, height) * 0.064 * props.zoom
  return [width * 0.53 + rx * scale, height * 0.49 + (ry * 0.50 - z) * scale] as const
}

function line(points: Point[], stroke: string, lineWidth: number) {
  if (!context) return
  context.beginPath()
  points.forEach(([x, y], index) => index ? context!.lineTo(x, y) : context!.moveTo(x, y))
  context.strokeStyle = stroke
  context.lineWidth = lineWidth
  context.stroke()
}

function drawCity() {
  if (!context) return
  const blocks = props.layout === 'tokyo' ? 9 : 8
  for (let i = -blocks; i <= blocks; i += 1) {
    const wiggle = props.layout === 'tokyo' ? Math.sin(i * 1.7) * 0.22 : 0
    line([project(i + wiggle, -9), project(i - wiggle, 9)], 'rgba(206,210,204,.13)', 1)
    line([project(-9, i - wiggle), project(9, i + wiggle)], 'rgba(206,210,204,.13)', 1)
  }
  if (props.layout === 'barcelona') {
    line([project(-9, -8), project(8, 9)], 'rgba(229,229,220,.22)', 4)
    line([project(-9, -8), project(8, 9)], 'rgba(8,9,8,.8)', 2)
  }
  for (let gx = -8; gx < 8; gx += 2) {
    for (let gy = -8; gy < 8; gy += 2) {
      if ((gx === -2 && gy === 0) || (gx === 0 && gy === 0)) continue
      const hash = Math.abs((gx * 92821 + gy * 68917) % 101)
      const bh = 0.48 + (hash / 100) * (hash % 5 === 0 ? 4.1 : 1.8)
      drawBuilding(gx + 0.16, gy + 0.16, 1.68, 1.68, bh)
    }
  }
  drawPark()
  drawTransit()
}

function drawBuilding(x: number, y: number, w: number, d: number, h: number) {
  if (!context) return
  const a = project(x, y), b = project(x + w, y), c = project(x + w, y + d), e = project(x, y + d)
  const at = project(x, y, h), bt = project(x + w, y, h), ct = project(x + w, y + d, h), et = project(x, y + d, h)
  context.fillStyle = 'rgba(172,177,170,.045)'
  context.beginPath(); context.moveTo(...at); context.lineTo(...bt); context.lineTo(...ct); context.lineTo(...et); context.closePath(); context.fill()
  line([a, b, c, e, a], 'rgba(215,219,211,.20)', .75)
  line([at, bt, ct, et, at], 'rgba(225,228,220,.29)', .8)
  line([a, at], 'rgba(215,219,211,.15)', .7); line([b, bt], 'rgba(215,219,211,.15)', .7)
  line([c, ct], 'rgba(215,219,211,.15)', .7); line([e, et], 'rgba(215,219,211,.15)', .7)
}

function drawPark() {
  if (!context) return
  const corners = [project(-1.85, .18), project(1.85, .18), project(1.85, 3.8), project(-1.85, 3.8)]
  context.beginPath(); context.moveTo(...corners[0]); corners.slice(1).forEach(p => context!.lineTo(...p)); context.closePath()
  context.fillStyle = 'rgba(167,182,158,.055)'; context.fill(); context.strokeStyle = 'rgba(196,213,186,.16)'; context.stroke()
  for (let i = 0; i < 13; i += 1) {
    const x = -1.55 + ((i * 43) % 30) / 10
    const y = .5 + ((i * 29) % 30) / 10
    const p = project(x, y, .18); context.beginPath(); context.arc(p[0], p[1], 2.2, 0, Math.PI * 2)
    context.fillStyle = 'rgba(184,200,175,.18)'; context.fill()
  }
}

function drawTransit() {
  line([project(-9, 5.2, .12), project(9, 5.2, .12)], 'rgba(224,222,211,.20)', 8)
  line([project(-9, 5.2, .14), project(9, 5.2, .14)], 'rgba(12,13,12,.92)', 4)
  line([project(-9, 5.15, .15), project(9, 5.15, .15)], 'rgba(217,216,205,.25)', 1)
  line([project(-9, -5.35, .4), project(9, -5.35, .4)], 'rgba(224,222,211,.16)', 12)
  line([project(-9, -5.35, .42), project(9, -5.35, .42)], 'rgba(11,12,11,.88)', 8)
}

function drawAgents(elapsed: number) {
  if (!context) return
  const colors = props.colorMode === 'emotion' ? AFFECT_COLORS : MODE_COLORS
  const scenarioRate = props.scenario === 'weekday' ? 1.12 : props.scenario === 'midday' ? .84 : .68
  const time = elapsed * (0.22 + props.agency * .003) * scenarioRate + props.minute * .035
  const count = Math.min(POPULATION, props.population)
  for (let i = 0; i < count; i += 1) {
    const lane = field.lane[i] % 18 - 9
    const horizontal = (field.lane[i] & 1) === 0
    const nativeMode = field.mode[i]
    const mode = nativeMode === 3 && (i % 100) >= props.transit ? 0 : nativeMode
    const rate = field.speed[i] * (mode === 2 ? 1.9 : mode === 1 ? .72 : 1)
    let along = ((field.phase[i] * 18 + time * rate) % 18) - 9
    let cross = lane + field.cross[i] * .075
    if (mode === 3) cross = 5.2 + field.cross[i] * .04
    if (mode === 4) cross = -5.35 + field.cross[i] * .05
    let x = horizontal ? along : cross
    let y = horizontal ? cross : along
    if (mode === 1 && (i % 11) < 5) {
      const angle = along * .82 + (i % 7)
      x = Math.cos(angle) * (1.05 + (i % 4) * .17)
      y = 2 + Math.sin(angle) * (1.05 + (i % 4) * .17)
    }
    const z = mode === 2 ? .42 : mode === 4 ? .65 + ((along + 9) / 18) * 2.3 : .08
    const p = project(x, y, z)
    if (p[0] < -2 || p[0] > width + 2 || p[1] < -2 || p[1] > height + 2) continue
    let affect = field.emotion[i]
    const rushHour = props.scenario === 'weekday' && props.minute >= 420 && props.minute <= 600
    if (mode === 1 && Math.abs(x) < 2 && y > 0 && y < 4) affect = (i & 1) === 0 ? 0 : 1
    else if (rushHour && (mode === 2 || mode === 3)) affect = field.phase[i] > .46 ? 3 : 2
    else if (props.scenario === 'weekend' && props.minute > 840) affect = 4
    context.fillStyle = colors[props.colorMode === 'emotion' ? affect : mode]
    const size = i % 19 === 0 ? 1.9 : 1.05
    context.globalAlpha = i % 13 === 0 ? .98 : .82
    context.fillRect(p[0], p[1], size, size)
  }
  context.globalAlpha = 1
}

function render(now: number) {
  if (!context) return
  context.setTransform(dpr, 0, 0, dpr, 0, 0)
  context.clearRect(0, 0, width, height)
  const delta = Math.min(.05, (now - lastFrame) / 1000)
  lastFrame = now
  if (!props.paused) simulationTime += delta
  drawCity(); drawAgents(simulationTime)
  animation = requestAnimationFrame(render)
}

function resize() {
  if (!canvas.value) return
  const rect = canvas.value.getBoundingClientRect()
  dpr = Math.min(devicePixelRatio, 1.5); width = rect.width; height = rect.height
  canvas.value.width = Math.round(width * dpr); canvas.value.height = Math.round(height * dpr)
}

function selectAgent(event: MouseEvent) {
  const rect = canvas.value!.getBoundingClientRect()
  const nx = (event.clientX - rect.left) / rect.width
  const ny = (event.clientY - rect.top) / rect.height
  emit('select', 1 + Math.floor(((nx * 73817 + ny * 19463) % 1) * (POPULATION - 1)))
}

watch(() => props.resetToken, () => { simulationTime = 0; lastFrame = performance.now() })
onMounted(() => {
  context = canvas.value!.getContext('2d', { alpha: true }); resize()
  window.addEventListener('resize', resize); animation = requestAnimationFrame(render)
})
onBeforeUnmount(() => { cancelAnimationFrame(animation); window.removeEventListener('resize', resize) })
</script>

<template><canvas ref="canvas" aria-label="Animated 3D city population simulation" @click="selectAgent" /></template>

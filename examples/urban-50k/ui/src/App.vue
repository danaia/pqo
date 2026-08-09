<script setup lang="ts">
import { computed, ref } from 'vue'
import CityCanvas from './CityCanvas.vue'
import { agentDetail, type ColorMode, type Layout } from './simulation'

const layout = ref<Layout>('manhattan')
const population = ref('50000')
const scenario = ref('weekday')
const agency = ref(84)
const transit = ref(61)
const colorMode = ref<ColorMode>('emotion')
const paused = ref(false)
const minute = ref(522)
const selectedId = ref(18472)
const zoom = ref(1)
const resetToken = ref(0)
const selected = computed(() => agentDetail(selectedId.value, minute.value))
const populationCount = computed(() => Number(population.value))
const activeCount = computed(() => Math.max(0, populationCount.value - 4).toLocaleString())
const inTransit = computed(() => Math.round(populationCount.value * .0012498 * transit.value).toLocaleString())
const meanSpeed = computed(() => scenario.value === 'weekday' ? '12.4' : scenario.value === 'midday' ? '10.8' : '9.2')
const clock = computed(() => {
  const hour = Math.floor(minute.value / 60) % 24
  const min = minute.value % 60
  return `${String(hour).padStart(2, '0')}:${String(min).padStart(2, '0')}:17`
})
const shortClock = computed(() => clock.value.slice(0, 5))

function reset() {
  paused.value = false
  minute.value = 522
  selectedId.value = 18472
  zoom.value = 1
  resetToken.value += 1
}
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <div class="brand">URBAN <span>/</span> 50K</div>
      <div class="product-name">Population mobility model</div>
      <div class="deterministic"><i /> DETERMINISTIC</div>
      <time>{{ clock }}</time>
      <div class="top-actions">
        <button class="outline-button" data-testid="pause" @click="paused = !paused">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path v-if="!paused" d="M8 5v14M16 5v14"/><path v-else d="m8 5 11 7-11 7Z"/></svg>
          {{ paused ? 'Resume' : 'Pause' }}
        </button>
        <button class="outline-button" data-testid="reset" @click="reset">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 7v5h-5M19 12a7 7 0 1 1-2-5"/></svg>
          Reset
        </button>
      </div>
    </header>

    <section class="stage">
      <CityCanvas
        :layout="layout" :color-mode="colorMode" :paused="paused" :minute="minute"
        :agency="agency" :transit="transit" :population="populationCount" :scenario="scenario"
        :zoom="zoom" :reset-token="resetToken"
        @select="selectedId = $event"
      />

      <aside class="control-panel light-panel">
        <label class="field-label" for="layout">CITY MODEL</label>
        <select id="layout" v-model="layout" data-testid="city-layout">
          <option value="manhattan">Manhattan Grid</option>
          <option value="barcelona">Barcelona Superblocks</option>
          <option value="tokyo">Tokyo Mixed Grid</option>
        </select>

        <label class="field-label" for="population">POPULATION</label>
        <select id="population" v-model="population">
          <option value="50000">50,000 agents</option>
          <option value="25000">25,000 agents</option>
          <option value="10000">10,000 agents</option>
        </select>

        <label class="field-label" for="scenario">BEHAVIOR</label>
        <select id="scenario" v-model="scenario">
          <option value="weekday">Weekday / AM peak</option>
          <option value="midday">Weekday / Midday</option>
          <option value="weekend">Weekend / Afternoon</option>
        </select>

        <label class="range-label" for="agency"><span>Agency</span><output>{{ agency }}%</output></label>
        <input id="agency" v-model.number="agency" type="range" min="0" max="100" />

        <label class="range-label" for="transit"><span>Transit reliance</span><output>{{ transit }}%</output></label>
        <input id="transit" v-model.number="transit" type="range" min="0" max="100" />

        <span class="field-label view-label">VIEW</span>
        <div class="segmented" role="group" aria-label="Particle color view">
          <button :class="{ active: colorMode === 'emotion' }" data-testid="emotion-view" @click="colorMode = 'emotion'">Emotion</button>
          <button :class="{ active: colorMode === 'mode' }" data-testid="mode-view" @click="colorMode = 'mode'">Mode</button>
        </div>
      </aside>

      <aside class="right-rail">
        <section class="legend light-panel">
          <h2>{{ colorMode === 'emotion' ? 'AFFECT STATE' : 'TRAVEL MODE' }}</h2>
          <ul v-if="colorMode === 'emotion'">
            <li><i class="calm" /> Calm</li><li><i class="content" /> Content</li><li><i class="alert" /> Alert</li>
            <li><i class="stressed" /> Stressed</li><li><i class="social" /> Social</li>
          </ul>
          <ul v-else>
            <li><i class="walk" /> Walk</li><li><i class="cycle" /> Run / Cycle</li><li><i class="car" /> Car</li>
            <li><i class="train" /> Train</li><li><i class="air" /> Air</li>
          </ul>
        </section>

        <section class="inspector light-panel" aria-live="polite">
          <header><h2>{{ selected.id }}</h2><button aria-label="Close agent inspector">×</button></header>
          <dl>
            <div><dt>intent</dt><dd>{{ selected.intent }}</dd></div>
            <div><dt>mode</dt><dd>{{ selected.mode }}</dd></div>
            <div><dt>next decision</dt><dd>{{ selected.next }}</dd></div>
            <div><dt>confidence</dt><dd>{{ selected.confidence }}%</dd></div>
          </dl>
        </section>
      </aside>

      <div class="camera-tools">
        <div class="scale"><span>100 m</span><i /></div>
        <div class="tool-row">
          <button aria-label="Reset camera" @click="zoom = 1"><svg viewBox="0 0 24 24"><path d="M12 3v3m0 12v3M3 12h3m12 0h3M7 7a7 7 0 1 0 10 0"/></svg></button>
          <button aria-label="Pan map"><svg viewBox="0 0 24 24"><path d="M8 11V6a1.5 1.5 0 0 1 3 0v5-7a1.5 1.5 0 0 1 3 0v7-5a1.5 1.5 0 0 1 3 0v7l1-1a1.6 1.6 0 0 1 2 2l-5 7H9l-5-6a1.5 1.5 0 0 1 2-2l2 2Z"/></svg></button>
          <button aria-label="Isometric view"><svg viewBox="0 0 24 24"><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9Zm0 0v9m8-4.5-8 4.5-8-4.5m8 4.5v9"/></svg></button>
          <button aria-label="Zoom out" @click="zoom = Math.max(.65, zoom - .1)">−</button>
          <button aria-label="Zoom in" @click="zoom = Math.min(1.5, zoom + .1)">+</button>
        </div>
      </div>

      <div class="metrics">
        <div><svg viewBox="0 0 24 24"><path d="M16 11a4 4 0 1 0-3.8-5.2M8 11a4 4 0 1 0 0-8m8 8c3.3 0 6 2 6 5v2h-7m-7-7c-3.3 0-6 2-6 5v2h12v-2c0-3-2.7-5-6-5Z"/></svg><strong>{{ activeCount }}</strong><span>active</span></div>
        <div><svg viewBox="0 0 24 24"><path d="M6 3h12a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2Zm0 15-2 3m14-3 2 3M7 7h10M8 14h.01M16 14h.01"/></svg><strong>{{ inTransit }}</strong><span>in transit</span></div>
        <div><svg viewBox="0 0 24 24"><path d="M4 18a8 8 0 1 1 16 0M12 18l4-6M6 18h12"/></svg><strong>{{ meanSpeed }}</strong><span>km/h mean</span></div>
      </div>

      <div class="timeline">
        <output :style="{ left: `${((minute - 360) / 900) * 100}%` }">{{ shortClock }}</output>
        <input v-model.number="minute" aria-label="Simulation time" type="range" min="360" max="1260" step="1" />
        <div class="time-labels"><span>06:00</span><span>09:00</span><span>12:00</span><span>15:00</span><span>18:00</span><span>21:00</span></div>
      </div>
    </section>
  </main>
</template>

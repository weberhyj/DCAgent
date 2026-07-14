<script setup lang="ts">
import { onBeforeUnmount, onMounted, useTemplateRef, watch } from 'vue'
import * as THREE from 'three'

const props = withDefaults(defineProps<{
  autoPulse?: boolean
}>(), {
  autoPulse: false,
})

const canvasRef = useTemplateRef<HTMLCanvasElement>('quantumCanvas')

const NODE_COUNT = 660
const EDGE_WEB_NODE_COUNT = 240
const MAX_CONNECTIONS = 2220
const KNOWLEDGE_DOMAIN_COUNT = 12
const KNOWLEDGE_RING_RADII = [0.82, 1.45, 2.12, 2.82, 3.48]
const EDGE_WEB_RING = KNOWLEDGE_RING_RADII.length + 1
const NETWORK_BASE_ROTATION_X = 0
const NETWORK_BASE_ROTATION_Y = 0
const NETWORK_BASE_ROTATION_Z = 0
const CAMERA_DISTANCE = 11.2
const DESKTOP_NETWORK_SCALE = 0.9
const MOBILE_NETWORK_SCALE = 0.42
const AUTO_PULSE_DURATION_MS = 3200
const INTRO_DURATION_SECONDS = 1.75
const INTRO_PULSE_DELAY_SECONDS = 0.3

let animationFrame = 0
let autoPulseTimer: number | undefined
let renderer: THREE.WebGLRenderer | undefined
let scene: THREE.Scene | undefined
let camera: THREE.PerspectiveCamera | undefined
let networkGroup: THREE.Group | undefined
let coreGroup: THREE.Group | undefined
let orbitGroup: THREE.Group | undefined
let pointsMaterial: THREE.ShaderMaterial | undefined
let linesMaterial: THREE.ShaderMaterial | undefined
let pointsGeometry: THREE.BufferGeometry | undefined
let linesGeometry: THREE.BufferGeometry | undefined
let resizeObserver: ResizeObserver | undefined
let clock: THREE.Clock | undefined
let reduceMotion = false

const raycaster = new THREE.Raycaster()
const pointer = new THREE.Vector2()
const pulsePlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0)
const pulsePlaneNormal = new THREE.Vector3(0, 0, 1)
const pulsePlaneOrigin = new THREE.Vector3(0, 0, 0)
const pulsePoint = new THREE.Vector3(0, 0, 0)
const localPulsePoint = new THREE.Vector3(0, 0, 0)
const autoPulsePoint = new THREE.Vector3(0, 0, 0)

const uniforms = {
  uTime: { value: 0 },
  uIntroProgress: { value: 0 },
  uIntroPulseProgress: { value: 0 },
  uPulseOrigin: { value: new THREE.Vector3(0, 0, 0) },
  uPulseTime: { value: -20 },
  uBaseColor: { value: new THREE.Color('#7dd3fc') },
  uAccentColor: { value: new THREE.Color('#a78bfa') },
  uHotColor: { value: new THREE.Color('#e8fbff') },
  uPixelRatio: { value: 1 },
}

const PULSE_IGNORE_SELECTOR = '[data-ignore-quantum-pulse]'

function shouldIgnorePulse(event: PointerEvent) {
  const target = event.target

  return target instanceof Element && Boolean(target.closest(PULSE_IGNORE_SELECTOR))
}

function prefersReducedMotion() {
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
}

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value))
}

function easeOutCubic(progress: number) {
  const inverted = 1 - clamp01(progress)

  return 1 - inverted * inverted * inverted
}

function seededRandom(seed: number) {
  let state = seed

  return () => {
    state |= 0
    state = (state + 0x6d2b79f5) | 0

    let value = Math.imul(state ^ (state >>> 15), 1 | state)
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value

    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

function createNodePositions() {
  const random = seededRandom(42)
  const positions = new Float32Array(NODE_COUNT * 3)
  const depths = new Float32Array(NODE_COUNT)
  const nodes: THREE.Vector3[] = []
  const domains = new Uint8Array(NODE_COUNT)
  const rings = new Uint8Array(NODE_COUNT)

  function addNode(vector: THREE.Vector3, depth: number, domain: number, ring: number) {
    const index = nodes.length

    nodes.push(vector)
    positions[index * 3] = vector.x
    positions[index * 3 + 1] = vector.y
    positions[index * 3 + 2] = vector.z
    depths[index] = depth
    domains[index] = domain
    rings[index] = ring
  }

  addNode(new THREE.Vector3(0, 0, 0), 0.5, 255, 0)

  for (let domain = 0; domain < KNOWLEDGE_DOMAIN_COUNT; domain += 1) {
    const angle = (domain / KNOWLEDGE_DOMAIN_COUNT) * Math.PI * 2
      - Math.PI / 2
      + Math.sin(domain * 1.73) * 0.11
      + Math.cos(domain * 0.91) * 0.055
    const anchorRadius = 0.58 + Math.sin(domain * 2.11) * 0.08

    addNode(
      new THREE.Vector3(
        Math.cos(angle) * anchorRadius * 1.18,
        Math.sin(angle) * anchorRadius * 0.9,
        Math.sin(domain * 1.7) * 0.035,
      ),
      0.16 + (domain / KNOWLEDGE_DOMAIN_COUNT) * 0.58,
      domain,
      0,
    )
  }

  let cursor = 0

  while (nodes.length < NODE_COUNT - EDGE_WEB_NODE_COUNT) {
    const domain = cursor % KNOWLEDGE_DOMAIN_COUNT
    const ring = 1 + Math.floor(cursor / KNOWLEDGE_DOMAIN_COUNT) % KNOWLEDGE_RING_RADII.length
    const layer = Math.floor(cursor / (KNOWLEDGE_DOMAIN_COUNT * KNOWLEDGE_RING_RADII.length))
    const ringRadius = KNOWLEDGE_RING_RADII[ring - 1]
    const domainWarp = Math.sin(domain * 1.77 + ring * 0.31) * 0.12 + Math.cos(domain * 0.64 + layer * 0.49) * 0.08
    const angle = (domain / KNOWLEDGE_DOMAIN_COUNT) * Math.PI * 2
      - Math.PI / 2
      + ring * 0.042
      + domainWarp
      + Math.sin(layer * 0.42 + ring * 0.8) * 0.06
      + (random() - 0.5) * (0.34 / Math.sqrt(ring))
    const radius = ringRadius
      + Math.sin(layer * 0.52 + domain * 0.42) * 0.14
      + Math.cos(domain * 1.2 + ring * 0.9) * 0.08
      + (random() - 0.5) * 0.28
    const threadDrift = Math.sin(layer * 0.72 + domain * 1.12) * 0.09
    const x = Math.cos(angle) * radius * 1.22 + Math.cos(angle + Math.PI / 2) * threadDrift
    const y = Math.sin(angle) * radius * 0.88 + Math.sin(angle + Math.PI / 2) * threadDrift
    const z = Math.sin(angle * 2.0 + layer * 0.44) * 0.055 + (random() - 0.5) * 0.045
    const depth = ring / (KNOWLEDGE_RING_RADII.length + 1) + layer * 0.012 + (random() - 0.5) * 0.035

    addNode(new THREE.Vector3(x, y, z), clamp01(depth), domain, ring)
    cursor += 1
  }

  const edgeWebs = [
    { x: 0, y: 4.05, width: 5.82, height: 1.32, depth: 0.82, spokes: 8 },
    { x: 0, y: -4.0, width: 5.96, height: 1.36, depth: 0.84, spokes: 8 },
    { x: -6.3, y: 0.02, width: 1.42, height: 3.28, depth: 0.9, spokes: 7 },
    { x: 6.34, y: -0.04, width: 1.46, height: 3.22, depth: 0.91, spokes: 7 },
    { x: -3.5, y: 2.48, width: 2.92, height: 1.52, depth: 0.64, spokes: 7 },
    { x: 3.48, y: 2.46, width: 3.05, height: 1.48, depth: 0.65, spokes: 7 },
    { x: -3.88, y: -2.66, width: 3.65, height: 1.82, depth: 0.66, spokes: 8 },
    { x: 3.62, y: -2.56, width: 3.18, height: 1.38, depth: 0.67, spokes: 7 },
  ]
  const edgeWebSequence = [0, 1, 4, 5, 6, 7, 0, 1, 4, 5, 6, 7, 2, 3]

  while (nodes.length < NODE_COUNT) {
    const edgeIndex = nodes.length - (NODE_COUNT - EDGE_WEB_NODE_COUNT)
    const side = edgeWebSequence[edgeIndex % edgeWebSequence.length]
    const web = edgeWebs[side]
    const strand = Math.floor(edgeIndex / edgeWebSequence.length)
    const spoke = strand % web.spokes
    const band = Math.floor(strand / web.spokes)
    const bridgePatch = side >= 4
    const angle = (spoke / web.spokes) * Math.PI * 2
      + Math.sin(strand * 0.57 + side) * 0.24
      + (random() - 0.5) * 0.32
    const radius = (bridgePatch ? 0.18 : 0.24) + (band % 6) * (bridgePatch ? 0.2 : 0.18) + (random() - 0.5) * 0.26
    const drift = Math.sin(strand * 1.11 + side * 0.8) * (bridgePatch ? 0.26 : 0.2)
    const x = web.x
      + Math.cos(angle) * web.width * radius
      + Math.cos(angle + Math.PI / 2) * drift
    const y = web.y
      + Math.sin(angle) * web.height * radius
      + Math.sin(angle + Math.PI / 2) * drift
    const z = Math.sin(angle * 2.0 + side) * 0.04 + (random() - 0.5) * 0.035
    const depth = web.depth + (band % 5) * 0.02 + (random() - 0.5) * 0.035

    addNode(new THREE.Vector3(x, y, z), clamp01(depth), side % KNOWLEDGE_DOMAIN_COUNT, EDGE_WEB_RING)
  }

  return { nodes, positions, depths, domains, rings }
}

function createConnectionAttributes(nodes: THREE.Vector3[], nodeDepths: Float32Array, domains: Uint8Array, rings: Uint8Array) {
  const positions: number[] = []
  const depths: number[] = []
  const degrees = new Array(nodes.length).fill(0) as number[]
  const connected = new Set<string>()

  function addConnection(from: number, to: number) {
    if (positions.length / 6 >= MAX_CONNECTIONS) return
    if (degrees[from] > 13 || degrees[to] > 13) return
    if (from === to) return

    const key = from < to ? `${from}:${to}` : `${to}:${from}`
    if (connected.has(key)) return

    const start = nodes[from]
    const end = nodes[to]

    positions.push(start.x, start.y, start.z, end.x, end.y, end.z)
    depths.push(nodeDepths[from], nodeDepths[to])
    degrees[from] += 1
    degrees[to] += 1
    connected.add(key)
  }

  for (let domain = 0; domain < KNOWLEDGE_DOMAIN_COUNT; domain += 1) {
    const anchorIndex = domain + 1
    const domainNodes = nodes
      .map((_, index) => index)
      .filter((index) => domains[index] === domain && index !== anchorIndex)
      .sort((left, right) => {
        if (rings[left] !== rings[right]) return rings[left] - rings[right]
        return nodes[left].lengthSq() - nodes[right].lengthSq()
      })

    addConnection(0, anchorIndex)

    for (let index = 0; index < Math.min(7, domainNodes.length); index += 1) {
      addConnection(anchorIndex, domainNodes[index])
    }

    for (let index = 0; index < domainNodes.length; index += 1) {
      const current = domainNodes[index]
      const next = domainNodes[index + 1]
      const skip = domainNodes[index + KNOWLEDGE_RING_RADII.length]

      if (next !== undefined && rings[current] === rings[next]) {
        addConnection(current, next)
      }

      if (skip !== undefined) {
        addConnection(current, skip)
      }
    }
  }

  for (let ring = 1; ring <= KNOWLEDGE_RING_RADII.length; ring += 1) {
    const ringNodes = nodes
      .map((_, index) => index)
      .filter((index) => rings[index] === ring)
      .sort((left, right) => Math.atan2(nodes[left].y, nodes[left].x) - Math.atan2(nodes[right].y, nodes[right].x))

    for (let index = 0; index < ringNodes.length; index += 1) {
      const current = ringNodes[index]
      const next = ringNodes[(index + 1) % ringNodes.length]
      const diagonal = ringNodes[(index + KNOWLEDGE_DOMAIN_COUNT + 1) % ringNodes.length]

      addConnection(current, next)

      if (index % 3 === 0) {
        addConnection(current, diagonal)
      }
    }
  }

  for (let lane = -6; lane <= 6; lane += 1) {
    const laneY = lane * 0.46 + Math.sin(lane * 1.4) * 0.14
    const laneNodes = nodes
      .map((_, index) => index)
      .filter((index) => {
        if (index === 0 || rings[index] === 0) return false

        const laneWindow = 0.24 + rings[index] * 0.045

        return Math.abs(nodes[index].y - laneY) < laneWindow
      })
      .sort((left, right) => nodes[left].x - nodes[right].x)

    for (let index = 0; index < laneNodes.length - 1; index += 1) {
      const current = laneNodes[index]
      const next = laneNodes[index + 1]
      const skip = laneNodes[index + 2]
      const xGap = Math.abs(nodes[current].x - nodes[next].x)
      const yGap = Math.abs(nodes[current].y - nodes[next].y)

      if (xGap < 1.52 && yGap < 0.42) {
        addConnection(current, next)
      }

      if (skip !== undefined && index % 3 === Math.abs(lane) % 3) {
        const skipXGap = Math.abs(nodes[current].x - nodes[skip].x)
        const skipYGap = Math.abs(nodes[current].y - nodes[skip].y)

        if (skipXGap < 2.18 && skipYGap < 0.5) {
          addConnection(current, skip)
        }
      }
    }
  }

  const edgeNodes = nodes
    .map((_, index) => index)
    .filter((index) => rings[index] === EDGE_WEB_RING)

  for (let index = 0; index < edgeNodes.length; index += 1) {
    const current = edgeNodes[index]

    for (let cursor = index + 1; cursor < edgeNodes.length; cursor += 1) {
      const candidate = edgeNodes[cursor]
      const distance = nodes[current].distanceTo(nodes[candidate])
      const sameSide = domains[current] === domains[candidate]
      const horizontalEdgeBand = domains[current] <= 1 && domains[candidate] <= 1
      const horizontalThread = Math.abs(nodes[current].y - nodes[candidate].y) < 0.36
      const verticalThread = Math.abs(nodes[current].x - nodes[candidate].x) < 0.42
      const sameSideDistanceLimit = horizontalEdgeBand ? 0.68 : 0.9
      const threadDistanceLimit = horizontalEdgeBand ? 1.12 : 1.65

      if (sameSide && distance < sameSideDistanceLimit) {
        addConnection(current, candidate)
      } else if ((horizontalThread || verticalThread) && distance < threadDistanceLimit && (index + cursor) % 7 === 0) {
        addConnection(current, candidate)
      }
    }

    const nearestOuter = nodes
      .map((_, candidate) => candidate)
      .filter((candidate) => rings[candidate] === KNOWLEDGE_RING_RADII.length)
      .sort((left, right) => nodes[current].distanceTo(nodes[left]) - nodes[current].distanceTo(nodes[right]))
      .slice(0, 1)

    nearestOuter.forEach((candidate) => {
      if (nodes[current].distanceTo(nodes[candidate]) < 2.85) {
        addConnection(current, candidate)
      }
    })

    const crossPatchBridge = edgeNodes
      .filter((candidate) => candidate !== current && domains[candidate] !== domains[current])
      .sort((left, right) => nodes[current].distanceTo(nodes[left]) - nodes[current].distanceTo(nodes[right]))
      .slice(0, 2)

    crossPatchBridge.forEach((candidate, bridgeIndex) => {
      const distance = nodes[current].distanceTo(nodes[candidate])
      const inBridgeField = Math.abs(nodes[current].x) < 5.7 && Math.abs(nodes[current].y) < 3.9
      const nearEnough = distance < (inBridgeField ? 1.86 : 1.34)
      const longOrganicStrand = inBridgeField && distance < 2.45 && (current + candidate + bridgeIndex) % 9 === 0

      if ((nearEnough && (index + bridgeIndex) % 2 === 0) || longOrganicStrand) {
        addConnection(current, candidate)
      }
    })
  }

  for (let index = 1; index < nodes.length; index += 1) {
    for (let cursor = index + 1; cursor < nodes.length; cursor += 1) {
      const sameDomain = domains[index] === domains[cursor]
      const adjacentRing = Math.abs(rings[index] - rings[cursor]) <= 1
      const crossDomainBridge = (index + cursor) % 37 === 0 && rings[index] >= 3 && rings[cursor] >= 3

      const distance = nodes[index].distanceTo(nodes[cursor])
      const depthGap = Math.abs(nodeDepths[index] - nodeDepths[cursor])

      if (sameDomain && adjacentRing && distance < 0.78 && depthGap < 0.22) {
        addConnection(index, cursor)
      } else if (!sameDomain && crossDomainBridge && distance < 1.42 && depthGap < 0.28) {
        addConnection(index, cursor)
      }
    }
  }

  return {
    positions: new Float32Array(positions),
    depths: new Float32Array(depths),
  }
}

function createPointMaterial() {
  return new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: `
      attribute float aDepth;
      varying float vAlpha;
      varying float vPulse;
      varying float vDepth;
      varying float vIntro;
      uniform float uTime;
      uniform float uIntroProgress;
      uniform float uIntroPulseProgress;
      uniform float uPulseTime;
      uniform float uPixelRatio;
      uniform vec3 uPulseOrigin;

      void main() {
        float stagedIntro = smoothstep(0.0, 1.0, clamp((uIntroProgress - aDepth * 0.18) / 0.82, 0.0, 1.0));
        float filamentAngle = aDepth * 42.0;
        vec3 folded = vec3(
          sin(filamentAngle) * 0.16,
          (aDepth - 0.5) * 1.36,
          cos(filamentAngle * 0.82) * 0.025
        );
        vec3 transformed = mix(folded, position, stagedIntro);
        float orbital = sin(uTime * 0.34 + transformed.y * 1.45 + aDepth * 8.0) * mix(0.025, 0.08, stagedIntro);
        transformed += normalize(transformed + vec3(0.01)) * orbital;

        float pulseAge = max(0.0, uTime - uPulseTime);
        float pulseDistance = distance(transformed.xy, uPulseOrigin.xy);
        float scenePulse = (1.0 - smoothstep(0.0, 0.58, abs(pulseDistance - pulseAge * 3.2))) * (1.0 - smoothstep(0.0, 5.4, pulseAge));
        float introPulseDistance = length(transformed.xy);
        float introPulseRadius = uIntroPulseProgress * 7.75;
        float introPulseBand = 1.0 - smoothstep(0.0, 0.66, abs(introPulseDistance - introPulseRadius));
        float introPulseEcho = (1.0 - smoothstep(0.0, 1.35, abs(introPulseDistance - introPulseRadius + 1.05))) * 0.12;
        float introPulseFade = smoothstep(0.0, 0.06, uIntroPulseProgress) * (1.0 - smoothstep(0.94, 1.0, uIntroPulseProgress));
        float introPulse = (introPulseBand * 0.58 + introPulseEcho) * introPulseFade;
        float introWave = (1.0 - smoothstep(0.0, 0.22, abs(aDepth - uIntroProgress))) * (1.0 - smoothstep(0.08, 0.98, uIntroProgress));
        vIntro = introWave * (1.0 - stagedIntro * 0.35);
        vPulse = max(max(scenePulse, introPulse), vIntro);
        vDepth = aDepth;

        vec4 mvPosition = modelViewMatrix * vec4(transformed, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        gl_PointSize = (2.4 + aDepth * 2.9 + vPulse * 8.8 + vIntro * 3.2) * uPixelRatio * (48.0 / max(3.8, -mvPosition.z));
        vAlpha = (0.34 + stagedIntro * 0.44) + vPulse * 0.5 + vIntro * 0.16;
      }
    `,
    fragmentShader: `
      varying float vAlpha;
      varying float vPulse;
      varying float vDepth;
      varying float vIntro;
      uniform vec3 uBaseColor;
      uniform vec3 uAccentColor;
      uniform vec3 uHotColor;

      void main() {
        vec2 centered = gl_PointCoord * 2.0 - 1.0;
        float radius = dot(centered, centered);
        if (radius > 1.0) discard;

        float core = 1.0 - smoothstep(0.0, 0.42, radius);
        float halo = (1.0 - smoothstep(0.0, 1.0, radius)) * 0.72;
        vec3 coldColor = mix(uBaseColor, uAccentColor, smoothstep(0.0, 1.0, vDepth));
        vec3 color = mix(coldColor, uHotColor, clamp(vPulse + vIntro * 0.55, 0.0, 1.0));
        float edgeFade = mix(1.0, 0.56, smoothstep(0.68, 1.0, vDepth));
        float alpha = (core + halo) * vAlpha * edgeFade;

        gl_FragColor = vec4(color, alpha);
      }
    `,
  })
}

function createLineMaterial() {
  return new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexShader: `
      attribute float aDepth;
      varying float vAlpha;
      varying float vPulse;
      varying float vDepth;
      varying float vIntro;
      uniform float uTime;
      uniform float uIntroProgress;
      uniform float uIntroPulseProgress;
      uniform float uPulseTime;
      uniform vec3 uPulseOrigin;

      void main() {
        float stagedIntro = smoothstep(0.0, 1.0, clamp((uIntroProgress - aDepth * 0.2) / 0.8, 0.0, 1.0));
        float filamentAngle = aDepth * 42.0;
        vec3 folded = vec3(
          sin(filamentAngle) * 0.16,
          (aDepth - 0.5) * 1.36,
          cos(filamentAngle * 0.82) * 0.025
        );
        vec3 transformed = mix(folded, position, stagedIntro);
        float drift = sin(uTime * 0.28 + transformed.x * 0.7 + transformed.y * 1.12) * mix(0.012, 0.045, stagedIntro);
        transformed += normalize(transformed + vec3(0.01)) * drift;

        float pulseAge = max(0.0, uTime - uPulseTime);
        float pulseDistance = distance(transformed.xy, uPulseOrigin.xy);
        float scenePulse = (1.0 - smoothstep(0.0, 0.7, abs(pulseDistance - pulseAge * 3.2))) * (1.0 - smoothstep(0.0, 5.2, pulseAge));
        float introPulseDistance = length(transformed.xy);
        float introPulseRadius = uIntroPulseProgress * 7.75;
        float introPulseBand = 1.0 - smoothstep(0.0, 0.76, abs(introPulseDistance - introPulseRadius));
        float introPulseEcho = (1.0 - smoothstep(0.0, 1.45, abs(introPulseDistance - introPulseRadius + 1.05))) * 0.1;
        float introPulseFade = smoothstep(0.0, 0.06, uIntroPulseProgress) * (1.0 - smoothstep(0.94, 1.0, uIntroPulseProgress));
        float introPulse = (introPulseBand * 0.68 + introPulseEcho) * introPulseFade;
        float introWave = (1.0 - smoothstep(0.0, 0.22, abs(aDepth - uIntroProgress))) * (1.0 - smoothstep(0.08, 0.98, uIntroProgress));
        vIntro = introWave * (1.0 - stagedIntro * 0.28);
        vPulse = max(max(scenePulse, introPulse), vIntro);
        vDepth = aDepth;
        vAlpha = (0.08 + stagedIntro * 0.2) + aDepth * 0.24 + vPulse * 0.56 + vIntro * 0.2;

        gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
      }
    `,
    fragmentShader: `
      varying float vAlpha;
      varying float vPulse;
      varying float vDepth;
      varying float vIntro;
      uniform vec3 uBaseColor;
      uniform vec3 uAccentColor;
      uniform vec3 uHotColor;

      void main() {
        vec3 coldColor = mix(uBaseColor, uAccentColor, smoothstep(0.0, 1.0, vDepth));
        vec3 color = mix(coldColor, uHotColor, clamp(vPulse + vIntro * 0.48, 0.0, 1.0));

        float edgeFade = mix(1.0, 0.52, smoothstep(0.68, 1.0, vDepth));

        gl_FragColor = vec4(color, vAlpha * edgeFade);
      }
    `,
  })
}

function createOrbitLine(radiusX: number, radiusY: number, zOffset: number, opacity: number) {
  const segments = 180
  const points: THREE.Vector3[] = []

  for (let index = 0; index <= segments; index += 1) {
    const angle = (index / segments) * Math.PI * 2
    const warp = 1
      + Math.sin(angle * 5.0 + radiusX * 0.72) * 0.035
      + Math.cos(angle * 8.0 + radiusY * 1.18) * 0.022
    const xTension = Math.cos(angle * 3.0 + radiusY) * 0.035
    const yTension = Math.sin(angle * 4.0 + radiusX) * 0.04

    points.push(new THREE.Vector3(
      Math.cos(angle) * radiusX * warp + xTension,
      Math.sin(angle) * radiusY * warp + yTension,
      zOffset + Math.sin(angle * 2.0 + radiusX) * 0.02,
    ))
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  const material = new THREE.LineBasicMaterial({
    color: 0x7dd3fc,
    transparent: true,
    opacity,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
  material.userData.baseOpacity = opacity

  return new THREE.Line(geometry, material)
}

function createRadialGuide(angle: number, length: number, opacity: number) {
  const bend = Math.sin(angle * 2.2) * 0.08
  const points = [
    new THREE.Vector3(Math.cos(angle) * 0.42, Math.sin(angle) * 0.38, 0),
    new THREE.Vector3(
      Math.cos(angle + bend) * length * 1.14,
      Math.sin(angle + bend) * length * 0.88,
      Math.sin(angle * 1.35) * 0.035,
    ),
  ]
  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  const material = new THREE.LineBasicMaterial({
    color: 0xb8f2ff,
    transparent: true,
    opacity,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
  material.userData.baseOpacity = opacity

  return new THREE.Line(geometry, material)
}

function createLateralThread(yOffset: number, width: number, phase: number, opacity: number) {
  const segments = 26
  const points: THREE.Vector3[] = []

  for (let index = 0; index <= segments; index += 1) {
    const progress = index / segments
    const x = (progress - 0.5) * width
    const y = yOffset
      + Math.sin(progress * Math.PI * 3.0 + phase) * 0.045
      + Math.cos(progress * Math.PI * 7.0 + phase * 0.6) * 0.022
    const z = Math.sin(progress * Math.PI * 2.0 + phase) * 0.018

    points.push(new THREE.Vector3(x, y, z))
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  const material = new THREE.LineBasicMaterial({
    color: 0x7dd3fc,
    transparent: true,
    opacity,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
  material.userData.baseOpacity = opacity

  return new THREE.Line(geometry, material)
}

function createVerticalThread(xOffset: number, height: number, phase: number, opacity: number) {
  const segments = 24
  const points: THREE.Vector3[] = []

  for (let index = 0; index <= segments; index += 1) {
    const progress = index / segments
    const x = xOffset
      + Math.sin(progress * Math.PI * 4.0 + phase) * 0.04
      + Math.cos(progress * Math.PI * 6.0 + phase * 0.7) * 0.024
    const y = (progress - 0.5) * height
    const z = Math.cos(progress * Math.PI * 2.0 + phase) * 0.016

    points.push(new THREE.Vector3(x, y, z))
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  const material = new THREE.LineBasicMaterial({
    color: 0x7dd3fc,
    transparent: true,
    opacity,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
  material.userData.baseOpacity = opacity

  return new THREE.Line(geometry, material)
}

function createBridgeThread(controlPoints: THREE.Vector3[], phase: number, opacity: number) {
  const curve = new THREE.CatmullRomCurve3(controlPoints)
  const segments = 42
  const points: THREE.Vector3[] = []

  for (let index = 0; index <= segments; index += 1) {
    const progress = index / segments
    const point = curve.getPoint(progress)
    const tangent = curve.getTangent(progress)
    const normal = new THREE.Vector3(-tangent.y, tangent.x, 0)

    if (normal.lengthSq() > 0) {
      normal.normalize()
    }

    const ripple = Math.sin(progress * Math.PI * 4.0 + phase) * 0.034
      + Math.cos(progress * Math.PI * 9.0 + phase * 0.7) * 0.014

    point.addScaledVector(normal, ripple)
    point.z += Math.sin(progress * Math.PI * 2.0 + phase) * 0.012
    points.push(point)
  }

  const geometry = new THREE.BufferGeometry().setFromPoints(points)
  const material = new THREE.LineBasicMaterial({
    color: 0x8edcff,
    transparent: true,
    opacity,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
  material.userData.baseOpacity = opacity

  return new THREE.Line(geometry, material)
}

function createKnowledgeOrbitGroup() {
  const group = new THREE.Group()

  KNOWLEDGE_RING_RADII.forEach((radius, index) => {
    const line = createOrbitLine(radius * 1.1, radius * 0.94, (index - 2) * 0.026, 0.16 - index * 0.018)

    group.add(line)
  })

  for (let domain = 0; domain < KNOWLEDGE_DOMAIN_COUNT; domain += 1) {
    const angle = (domain / KNOWLEDGE_DOMAIN_COUNT) * Math.PI * 2
      - Math.PI * 0.08
      + Math.sin(domain * 1.37) * 0.075

    group.add(createRadialGuide(angle, 3.5, 0.11))
  }

  const lateralThreadLanes = [-1.25, 0.85]

  lateralThreadLanes.forEach((lane, index) => {
    const yOffset = lane * 0.5 + Math.sin(lane * 1.2) * 0.12
    const width = 9.86 - Math.abs(lane) * 0.42
    const opacity = 0.034 - Math.abs(lane) * 0.004

    group.add(createLateralThread(yOffset, width, lane * 0.82 + index * 0.18, opacity))
  })

  ;[-6.04, 6.04].forEach((xOffset, index) => {
    group.add(createVerticalThread(xOffset, 5.84, index * 1.2 + 0.5, 0.016))
  })

  const bridgeThreads = [
    [new THREE.Vector3(-5.55, 1.26, 0.01), new THREE.Vector3(-3.92, 2.7, -0.01), new THREE.Vector3(-1.18, 2.18, 0.02)],
    [new THREE.Vector3(-5.2, 0.82, -0.01), new THREE.Vector3(-3.0, 1.72, 0.02), new THREE.Vector3(-0.72, 1.42, -0.01)],
    [new THREE.Vector3(1.05, 2.12, 0.02), new THREE.Vector3(3.52, 2.74, -0.01), new THREE.Vector3(5.55, 1.38, 0.01)],
    [new THREE.Vector3(0.82, 1.38, -0.01), new THREE.Vector3(3.1, 1.72, 0.01), new THREE.Vector3(5.25, 0.82, -0.02)],
    [new THREE.Vector3(-5.5, -1.32, 0.02), new THREE.Vector3(-3.7, -2.58, -0.01), new THREE.Vector3(-1.18, -2.02, 0.01)],
    [new THREE.Vector3(-5.18, -0.88, -0.01), new THREE.Vector3(-3.08, -1.72, 0.02), new THREE.Vector3(-0.82, -1.34, -0.01)],
    [new THREE.Vector3(-5.18, -3.02, 0.01), new THREE.Vector3(-3.55, -3.36, -0.02), new THREE.Vector3(-1.05, -2.72, 0.02)],
    [new THREE.Vector3(1.12, -2.04, 0.01), new THREE.Vector3(3.48, -2.62, -0.02), new THREE.Vector3(5.55, -1.35, 0.02)],
    [new THREE.Vector3(0.86, -1.32, -0.02), new THREE.Vector3(3.12, -1.72, 0.01), new THREE.Vector3(5.22, -0.86, -0.01)],
  ]

  bridgeThreads.forEach((points, index) => {
    group.add(createBridgeThread(points, index * 0.73 + 0.2, 0.026))
  })

  return group
}

function createAgentCoreGroup() {
  const group = new THREE.Group()
  const coreGeometry = new THREE.IcosahedronGeometry(0.42, 1)
  const coreMaterial = new THREE.MeshBasicMaterial({
    color: 0xe8fbff,
    transparent: true,
    opacity: 0.54,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
  coreMaterial.userData.baseOpacity = coreMaterial.opacity
  const core = new THREE.Mesh(coreGeometry, coreMaterial)
  const shellGeometry = new THREE.IcosahedronGeometry(0.68, 1)
  const shellMaterial = new THREE.MeshBasicMaterial({
    color: 0x67d8ff,
    transparent: true,
    opacity: 0.22,
    wireframe: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
  shellMaterial.userData.baseOpacity = shellMaterial.opacity
  const shell = new THREE.Mesh(shellGeometry, shellMaterial)
  const haloGeometry = new THREE.TorusGeometry(0.86, 0.008, 8, 120)
  const haloMaterial = new THREE.MeshBasicMaterial({
    color: 0xb8f2ff,
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
  haloMaterial.userData.baseOpacity = haloMaterial.opacity
  const halo = new THREE.Mesh(haloGeometry, haloMaterial)
  const tiltedHalo = new THREE.Mesh(haloGeometry.clone(), haloMaterial.clone())
  ;(tiltedHalo.material as THREE.Material).userData.baseOpacity = haloMaterial.opacity

  halo.rotation.x = Math.PI * 0.52
  tiltedHalo.rotation.y = Math.PI * 0.42
  tiltedHalo.rotation.x = Math.PI * 0.18

  group.add(halo, tiltedHalo, shell, core)

  return group
}

function setGroupOpacity(group: THREE.Group | undefined, multiplier: number) {
  group?.traverse((object) => {
    const item = object as THREE.Object3D & {
      material?: THREE.Material | THREE.Material[]
    }
    const materials = Array.isArray(item.material) ? item.material : item.material ? [item.material] : []

    materials.forEach((material) => {
      const baseOpacity = typeof material.userData.baseOpacity === 'number' ? material.userData.baseOpacity : material.opacity

      material.opacity = baseOpacity * multiplier
    })
  })
}

function disposeMaterial(material: THREE.Material | THREE.Material[]) {
  if (Array.isArray(material)) {
    material.forEach((entry) => entry.dispose())
    return
  }

  material.dispose()
}

function disposeGroupAssets(group: THREE.Group | undefined) {
  group?.traverse((object) => {
    const item = object as THREE.Object3D & {
      geometry?: THREE.BufferGeometry
      material?: THREE.Material | THREE.Material[]
    }

    item.geometry?.dispose()

    if (item.material) {
      disposeMaterial(item.material)
    }
  })
}

function setupScene(canvas: HTMLCanvasElement) {
  const { nodes, positions, depths, domains, rings } = createNodePositions()
  const connections = createConnectionAttributes(nodes, depths, domains, rings)

  scene = new THREE.Scene()
  scene.fog = new THREE.FogExp2(0x071017, 0.032)

  camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100)
  camera.position.set(0, 0, CAMERA_DISTANCE)
  camera.lookAt(0, 0, 0)

  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  })
  renderer.setClearColor(0x050508, 0)

  networkGroup = new THREE.Group()
  networkGroup.rotation.set(NETWORK_BASE_ROTATION_X, NETWORK_BASE_ROTATION_Y, NETWORK_BASE_ROTATION_Z)
  scene.add(networkGroup)

  orbitGroup = createKnowledgeOrbitGroup()
  networkGroup.add(orbitGroup)

  pointsGeometry = new THREE.BufferGeometry()
  pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  pointsGeometry.setAttribute('aDepth', new THREE.BufferAttribute(depths, 1))

  linesGeometry = new THREE.BufferGeometry()
  linesGeometry.setAttribute('position', new THREE.BufferAttribute(connections.positions, 3))
  linesGeometry.setAttribute('aDepth', new THREE.BufferAttribute(connections.depths, 1))

  pointsMaterial = createPointMaterial()
  linesMaterial = createLineMaterial()

  networkGroup.add(new THREE.LineSegments(linesGeometry, linesMaterial))
  networkGroup.add(new THREE.Points(pointsGeometry, pointsMaterial))

  coreGroup = createAgentCoreGroup()
  networkGroup.add(coreGroup)

  clock = new THREE.Clock()
}

function resize() {
  if (!renderer || !camera || !canvasRef.value || !networkGroup) return

  const { width, height } = canvasRef.value.getBoundingClientRect()
  const safeWidth = Math.max(1, width)
  const safeHeight = Math.max(1, height)
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2)

  renderer.setPixelRatio(pixelRatio)
  renderer.setSize(safeWidth, safeHeight, false)
  camera.aspect = safeWidth / safeHeight

  if (safeWidth > 920) {
    networkGroup.position.set(0, 0, -0.24)
    networkGroup.scale.set(DESKTOP_NETWORK_SCALE * 1.18, DESKTOP_NETWORK_SCALE * 1.05, DESKTOP_NETWORK_SCALE)
  } else {
    networkGroup.position.set(0, 0, -0.38)
    networkGroup.scale.set(MOBILE_NETWORK_SCALE * 0.92, MOBILE_NETWORK_SCALE, MOBILE_NETWORK_SCALE)
  }

  camera.updateProjectionMatrix()
  uniforms.uPixelRatio.value = pixelRatio
}

function updatePulsePlane() {
  if (!networkGroup) return

  pulsePlaneNormal.set(0, 0, 1).applyQuaternion(networkGroup.quaternion).normalize()
  networkGroup.getWorldPosition(pulsePlaneOrigin)
  pulsePlane.setFromNormalAndCoplanarPoint(pulsePlaneNormal, pulsePlaneOrigin)
}

function triggerScenePulse(origin: THREE.Vector3) {
  if (!clock) return

  uniforms.uPulseOrigin.value.copy(origin)
  uniforms.uPulseTime.value = clock.getElapsedTime()
}

function triggerAutoPulse() {
  if (!clock) return

  autoPulsePoint.set(0, 0, 0)
  triggerScenePulse(autoPulsePoint)
}

function startAutoPulse() {
  if (autoPulseTimer !== undefined || prefersReducedMotion()) return

  triggerAutoPulse()
  autoPulseTimer = window.setTimeout(() => {
    autoPulseTimer = undefined
    if (props.autoPulse) {
      startAutoPulse()
    }
  }, AUTO_PULSE_DURATION_MS)
}

function stopAutoPulse() {
  if (autoPulseTimer === undefined) return

  window.clearTimeout(autoPulseTimer)
  autoPulseTimer = undefined
}

function triggerPulse(event: PointerEvent) {
  if (shouldIgnorePulse(event)) return
  if (!canvasRef.value || !camera || !clock) return

  const rect = canvasRef.value.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top

  if (x < 0 || x > rect.width || y < 0 || y > rect.height) return

  pointer.x = (x / rect.width) * 2 - 1
  pointer.y = -(y / rect.height) * 2 + 1
  raycaster.setFromCamera(pointer, camera)
  updatePulsePlane()

  if (raycaster.ray.intersectPlane(pulsePlane, pulsePoint)) {
    localPulsePoint.copy(pulsePoint)
    networkGroup?.worldToLocal(localPulsePoint)
    triggerScenePulse(localPulsePoint)
  }
}

function render() {
  if (!renderer || !scene || !camera || !networkGroup || !clock) return

  const elapsed = clock.getElapsedTime()
  const introPulseProgress = reduceMotion ? 1 : clamp01((elapsed - INTRO_PULSE_DELAY_SECONDS) / INTRO_DURATION_SECONDS)
  const introProgress = reduceMotion ? 1 : easeOutCubic(elapsed / INTRO_DURATION_SECONDS)

  uniforms.uTime.value = elapsed
  uniforms.uIntroProgress.value = introProgress
  uniforms.uIntroPulseProgress.value = introPulseProgress

  if (coreGroup) {
    const corePulse = 1 + Math.sin(elapsed * 1.8) * 0.035
    const coreIntroScale = 0.52 + introProgress * 0.48

    coreGroup.rotation.x += 0.0028
    coreGroup.rotation.y += 0.006
    coreGroup.scale.setScalar(coreIntroScale * corePulse)
    setGroupOpacity(coreGroup, 0.42 + introProgress * 0.58)
  }

  if (orbitGroup) {
    orbitGroup.scale.setScalar(0.22 + introProgress * 0.78)
    setGroupOpacity(orbitGroup, 0.08 + introProgress * 0.92)
  }

  renderer.render(scene, camera)
  animationFrame = window.requestAnimationFrame(render)
}

function disposeThree() {
  window.cancelAnimationFrame(animationFrame)
  window.removeEventListener('pointerdown', triggerPulse)
  stopAutoPulse()
  resizeObserver?.disconnect()

  disposeGroupAssets(coreGroup)
  disposeGroupAssets(orbitGroup)
  pointsGeometry?.dispose()
  linesGeometry?.dispose()
  pointsMaterial?.dispose()
  linesMaterial?.dispose()
  renderer?.dispose()

  renderer = undefined
  scene = undefined
  camera = undefined
  networkGroup = undefined
  coreGroup = undefined
  orbitGroup = undefined
  uniforms.uIntroProgress.value = 0
  uniforms.uIntroPulseProgress.value = 0
}

watch(() => props.autoPulse, (autoPulse) => {
  if (autoPulse) {
    startAutoPulse()
    return
  }

  stopAutoPulse()
})

onMounted(() => {
  const canvas = canvasRef.value
  if (!canvas) return

  reduceMotion = prefersReducedMotion()
  setupScene(canvas)
  resizeObserver = new ResizeObserver(resize)
  resizeObserver.observe(canvas)
  window.addEventListener('pointerdown', triggerPulse, { passive: true })
  resize()
  if (props.autoPulse) {
    startAutoPulse()
  }
  render()
})

onBeforeUnmount(() => {
  disposeThree()
})
</script>

<template>
  <div class="quantum-network" aria-hidden="true">
    <canvas ref="quantumCanvas" class="quantum-canvas" />
    <div class="quantum-vignette" />
    <div class="quantum-focus-shield" />
  </div>
</template>

<style scoped>
.quantum-network {
  position: absolute;
  inset: 0;
  overflow: hidden;
  background:
    radial-gradient(circle at 70% 46%, rgba(103, 216, 255, 0.28), transparent 38%),
    radial-gradient(circle at 42% 12%, rgba(167, 139, 250, 0.16), transparent 30%),
    #071017;
  pointer-events: none;
}

.quantum-canvas,
.quantum-vignette,
.quantum-focus-shield {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

.quantum-canvas {
  display: block;
  opacity: 1;
  filter: brightness(1.28) saturate(1.24) contrast(1.08);
}

.quantum-vignette {
  background:
    linear-gradient(90deg, rgba(7, 16, 23, 0.42) 0%, rgba(7, 16, 23, 0.13) 30%, rgba(7, 16, 23, 0.02) 100%),
    radial-gradient(ellipse at 72% 50%, transparent 0%, rgba(7, 16, 23, 0.035) 44%, rgba(7, 16, 23, 0.22) 100%),
    repeating-linear-gradient(180deg, rgba(255, 255, 255, 0.018) 0 1px, transparent 1px 6px);
  mix-blend-mode: normal;
}

.quantum-focus-shield {
  background:
    radial-gradient(ellipse 44% 24% at 50% 51%, rgba(5, 13, 19, 0.5) 0%, rgba(6, 15, 22, 0.42) 38%, rgba(7, 16, 23, 0.18) 66%, transparent 100%),
    radial-gradient(ellipse 36% 16% at 50% 52%, rgba(2, 8, 12, 0.36) 0%, rgba(2, 8, 12, 0.18) 48%, transparent 100%);
  opacity: 0.86;
}

@media (max-width: 920px) {
  .quantum-canvas {
    opacity: 0.92;
  }

  .quantum-vignette {
    background:
      linear-gradient(180deg, rgba(7, 16, 23, 0.46) 0%, rgba(7, 16, 23, 0.14) 42%, rgba(7, 16, 23, 0.38) 100%),
      repeating-linear-gradient(180deg, rgba(255, 255, 255, 0.018) 0 1px, transparent 1px 6px);
  }

  .quantum-focus-shield {
    background:
      radial-gradient(ellipse 70% 32% at 50% 50%, rgba(5, 13, 19, 0.52) 0%, rgba(6, 15, 22, 0.38) 45%, transparent 100%);
    opacity: 0.78;
  }
}
</style>

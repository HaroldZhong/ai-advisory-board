import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import tailwindcss from 'tailwindcss'
import autoprefixer from 'autoprefixer'

const MARKDOWN_PACKAGES = new Set([
  'bail',
  'ccount',
  'character-entities',
  'comma-separated-tokens',
  'decode-named-character-reference',
  'devlop',
  'html-url-attributes',
  'is-plain-obj',
  'katex',
  'longest-streak',
  'property-information',
  'react-markdown',
  'rehype-katex',
  'remark-math',
  'space-separated-tokens',
  'trim-lines',
  'trough',
  'unified',
  'vfile',
  'vfile-message',
  'web-namespaces',
  'zwitch',
])

const MARKDOWN_PACKAGE_PREFIXES = [
  'hast-util-',
  'mdast-util-',
  'micromark',
  'unist-util-',
]

function getPackageName(id) {
  const normalized = id.replace(/\\/g, '/')
  const nodeModulesIndex = normalized.lastIndexOf('/node_modules/')
  if (nodeModulesIndex === -1) return null

  const packagePath = normalized.slice(nodeModulesIndex + '/node_modules/'.length)
  const segments = packagePath.split('/')

  if (segments[0]?.startsWith('@')) {
    return `${segments[0]}/${segments[1]}`
  }

  return segments[0]
}

function manualChunks(id) {
  const packageName = getPackageName(id)
  if (!packageName) return undefined

  if (['react', 'react-dom', 'react-router', 'react-router-dom', 'scheduler'].includes(packageName)) {
    return 'vendor-react'
  }

  if (
    packageName.startsWith('@floating-ui/') ||
    packageName.startsWith('@radix-ui/') ||
    ['class-variance-authority', 'clsx', 'lucide-react', 'tailwind-merge'].includes(packageName)
  ) {
    return 'vendor-ui'
  }

  if (
    MARKDOWN_PACKAGES.has(packageName) ||
    MARKDOWN_PACKAGE_PREFIXES.some((prefix) => packageName.startsWith(prefix))
  ) {
    return 'vendor-markdown'
  }

  return 'vendor'
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  css: {
    postcss: {
      plugins: [
        tailwindcss,
        autoprefixer,
      ],
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks,
      },
    },
  },
})

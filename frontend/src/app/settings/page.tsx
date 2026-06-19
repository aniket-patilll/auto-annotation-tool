'use client'
import { useEffect, useState } from 'react'
import Link from 'next/link'

type Settings = {
  classes_dir: string
  output_base_dir: string
  default_labeler: string
  default_yolo_model: string
  roboflow_api_key: string
  roboflow_workspace: string
}

type RoboflowStatus = {
  ok: boolean
  workspace_name?: string
  project_count?: number
  projects?: string[]
  error?: string
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>({
    classes_dir: '', output_base_dir: '', default_labeler: 'yolo',
    default_yolo_model: '', roboflow_api_key: '', roboflow_workspace: '',
  })
  const [models, setModels]             = useState<{ name: string; path: string }[]>([])
  const [saved, setSaved]               = useState(false)
  const [rfStatus, setRfStatus]         = useState<RoboflowStatus | null>(null)
  const [rfTesting, setRfTesting]       = useState(false)
  const [rfProjects, setRfProjects]     = useState<any[]>([])
  const [loadingProjects, setLoadingProjects] = useState(false)

  useEffect(() => {
    fetch('/api/settings').then(r => r.json()).then(setSettings)
    fetch('/api/models').then(r => r.json()).then(d => setModels(d.models ?? []))
  }, [])

  async function save() {
    const r = await fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    })
    if (r.ok) {
      setSettings(await r.json())
      setSaved(true)
      setTimeout(() => setSaved(false), 2500)
    }
  }

  async function handleBrowse(key: keyof Settings, mode: 'folder' | 'file', fileTypes?: string) {
    const current = settings[key] || ''
    const qs = new URLSearchParams({ mode, start: current, file_types: fileTypes || '' })
    try {
      const r = await fetch(`/api/browse?${qs.toString()}`)
      if (r.ok) {
        const { path } = await r.json()
        if (path) setSettings(s => ({ ...s, [key]: path }))
      }
    } catch (err) { console.error('Failed to browse', err) }
  }

  async function testRoboflow() {
    setRfTesting(true)
    setRfStatus(null)
    // save first so the backend can read the key
    await fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    })
    try {
      const r = await fetch('/api/roboflow/validate')
      setRfStatus(await r.json())
    } catch (e) {
      setRfStatus({ ok: false, error: String(e) })
    } finally {
      setRfTesting(false)
    }
  }

  async function loadProjects() {
    setLoadingProjects(true)
    try {
      const r = await fetch('/api/roboflow/projects')
      const d = await r.json()
      setRfProjects(d.projects ?? [])
    } catch { }
    setLoadingProjects(false)
  }

  const field = (
    label: string,
    key: keyof Settings,
    placeholder = '',
    browseMode?: 'folder' | 'file',
    fileTypes?: string,
    isPassword = false,
  ) => (
    <div className="mb-4">
      <label className="block text-[11px] text-slate-500 uppercase tracking-widest mb-1.5">{label}</label>
      <div className="flex gap-2">
        <input
          type={isPassword ? 'password' : 'text'}
          value={settings[key]}
          onChange={e => setSettings(s => ({ ...s, [key]: e.target.value }))}
          className="flex-1 bg-[#080810] border border-[#1e2035] rounded-md px-3 py-2 text-sm text-slate-300 focus:outline-none focus:border-indigo-500 font-mono"
          placeholder={placeholder}
        />
        {browseMode && (
          <button
            onClick={() => handleBrowse(key, browseMode, fileTypes)}
            type="button"
            className="bg-[#1e2035] hover:bg-[#2a2a45] text-slate-300 border border-[#1e2035] px-4 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap">
            Browse…
          </button>
        )}
      </div>
    </div>
  )

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <Link href="/"
          className="text-sm bg-[#1e2035] hover:bg-[#2a2a45] text-slate-300 px-3 py-1.5 rounded-md transition-colors">
          ← Back
        </Link>
        <h1 className="text-base font-semibold text-slate-100">Settings</h1>
      </div>

      {/* ── Storage ────────────────────────────────────────────────────────── */}
      <div className="bg-[#13131f] border border-[#1e2035] rounded-xl p-6 mb-4">
        <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-widest mb-4">Storage</div>
        {field('Annotations save directory', 'classes_dir', 'D:\\yolo_world_poc\\annotatted_classes', 'folder')}
        {field('Job outputs base directory', 'output_base_dir', 'D:\\yolo_world_poc\\annotation_jobs', 'folder')}
      </div>

      {/* ── Default Model ──────────────────────────────────────────────────── */}
      <div className="bg-[#13131f] border border-[#1e2035] rounded-xl p-6 mb-4">
        <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-widest mb-4">Default Local Model</div>

        <div className="mb-4">
          <label className="block text-[11px] text-slate-500 uppercase tracking-widest mb-1.5">Default labeler</label>
          <select value={settings.default_labeler}
            onChange={e => setSettings(s => ({ ...s, default_labeler: e.target.value }))}
            className="w-full bg-[#080810] border border-[#1e2035] rounded-md px-3 py-2 text-sm text-slate-300 focus:outline-none focus:border-indigo-500">
            <option value="yolo">YOLO (custom model)</option>
            <option value="yolo_world">YOLO-World (open-vocab)</option>
            <option value="grounding_dino">Grounding DINO</option>
            <option value="roboflow">Roboflow Cloud ☁️</option>
          </select>
        </div>

        {field('Default YOLO / YOLO-World model path (.pt)', 'default_yolo_model',
          'D:\\yolo_world_poc\\models\\best.pt', 'file', 'pt')}

        {models.length > 0 && (
          <div className="mb-2">
            <div className="text-[11px] text-slate-600 mb-2">Detected model files:</div>
            <div className="flex flex-col gap-1.5">
              {models.map(m => (
                <button key={m.path}
                  onClick={() => setSettings(s => ({ ...s, default_yolo_model: m.path }))}
                  className="text-left text-xs text-indigo-400 hover:text-indigo-300 bg-indigo-950/20 border border-indigo-900/30 px-3 py-1.5 rounded-md transition-colors">
                  {m.name}
                  <span className="text-slate-600 ml-2 text-[10px]">{m.path}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── Roboflow Cloud ────────────────────────────────────────────────── */}
      <div className="bg-[#13131f] border border-[#1e2035] rounded-xl p-6 mb-4 relative overflow-hidden">
        {/* accent bar */}
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-purple-600 via-violet-500 to-indigo-500" />

        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-[11px] font-semibold text-slate-500 uppercase tracking-widest">Roboflow Cloud ☁️</div>
            <div className="text-xs text-slate-600 mt-0.5">Train models in Roboflow — use them here for auto-labeling</div>
          </div>
          {rfStatus && (
            <div className={`text-xs px-2.5 py-1 rounded-full font-medium ${rfStatus.ok
              ? 'bg-emerald-900/40 text-emerald-400 border border-emerald-800/50'
              : 'bg-red-900/40 text-red-400 border border-red-800/50'
            }`}>
              {rfStatus.ok ? `✓ Connected — ${rfStatus.project_count} projects` : `✗ ${rfStatus.error}`}
            </div>
          )}
        </div>

        {/* Info box */}
        <div className="bg-violet-950/20 border border-violet-900/30 rounded-lg p-3 mb-4 text-xs text-slate-400 leading-relaxed">
          <div className="text-violet-300 font-medium mb-1">How to use:</div>
          <ol className="list-decimal list-inside space-y-0.5">
            <li>Upload frames to Roboflow → annotate → train model there</li>
            <li>Enter your API key + workspace slug below</li>
            <li>When creating a new job, select <span className="text-violet-300 font-medium">Roboflow Cloud</span> as labeler</li>
            <li>Enter your model ID (e.g. <code className="bg-[#0a0a15] px-1 py-0.5 rounded text-violet-200">my-project/3</code>)</li>
            <li>The tool calls Roboflow's cloud GPU — no local GPU needed!</li>
          </ol>
        </div>

        {field('API Key', 'roboflow_api_key', 'Your Roboflow API key', undefined, undefined, true)}
        {field('Workspace Slug', 'roboflow_workspace', 'your-workspace-slug (from app.roboflow.com URL)')}

        <div className="flex gap-2 mt-2 flex-wrap">
          <button
            onClick={testRoboflow}
            disabled={rfTesting || !settings.roboflow_api_key}
            className="flex items-center gap-2 bg-violet-700 hover:bg-violet-600 disabled:bg-[#1e2035] disabled:text-slate-600 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors">
            {rfTesting ? (
              <><span className="animate-spin inline-block w-3 h-3 border border-white/30 border-t-white rounded-full" />Testing…</>
            ) : '⚡ Test Connection'}
          </button>
          {rfStatus?.ok && (
            <button
              onClick={loadProjects}
              disabled={loadingProjects}
              className="bg-[#1e2035] hover:bg-[#2a2a45] text-slate-300 text-sm font-medium px-4 py-2 rounded-lg transition-colors">
              {loadingProjects ? 'Loading…' : '📋 Browse Projects'}
            </button>
          )}
          <a href="https://app.roboflow.com" target="_blank" rel="noopener noreferrer"
            className="bg-[#1e2035] hover:bg-[#2a2a45] text-slate-400 text-sm px-4 py-2 rounded-lg transition-colors">
            Open Roboflow ↗
          </a>
        </div>

        {/* Project list */}
        {rfProjects.length > 0 && (
          <div className="mt-4">
            <div className="text-[10px] text-slate-600 uppercase tracking-widest mb-2">Your Roboflow Projects</div>
            <div className="grid grid-cols-2 gap-1.5">
              {rfProjects.map(p => (
                <div key={p.id} className="bg-[#0d0d1a] border border-[#1e2035] rounded-md px-3 py-2">
                  <div className="text-xs text-slate-300 font-medium truncate">{p.name}</div>
                  <div className="text-[10px] text-slate-600 mt-0.5 flex items-center gap-1.5">
                    <code className="text-violet-400">{p.id}</code>
                    {p.versions > 0 && <span className="text-emerald-600">· {p.versions} version{p.versions > 1 ? 's' : ''}</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── Save ──────────────────────────────────────────────────────────────── */}
      <button onClick={save}
        className="w-full bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium px-6 py-2.5 rounded-lg transition-colors">
        {saved ? '✓ Settings Saved!' : 'Save Settings'}
      </button>
    </div>
  )
}

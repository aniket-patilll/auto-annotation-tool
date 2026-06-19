'use client'
import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'

type Job = {
  id: string; name: string; status: 'queued' | 'running' | 'ready' | 'error'
  saved: number; skipped: number; created_at: string
}
type LogEntry = { text: string; type: 'ok' | 'err' | 'default' }

const STATUS_CLS = {
  queued:  'bg-amber-950/60 text-amber-400 border border-amber-900/40 animate-pulse',
  running: 'bg-indigo-950/60 text-indigo-400 border border-indigo-900/40 shadow-[0_0_15px_rgba(99,102,241,0.15)] animate-pulse',
  ready:   'bg-emerald-950/60 text-emerald-400 border border-emerald-900/40',
  error:   'bg-rose-950/60 text-rose-400 border border-rose-900/40',
}

export default function Home() {
  const router = useRouter()
  const [jobs, setJobs]                     = useState<Job[]>([])
  const [tab, setTab]                       = useState<'new' | 'existing'>('new')
  const [inputMode, setInputMode]           = useState<'video_path' | 'upload' | 'images_path' | 'image'>('video_path')
  const [labeler, setLabeler]               = useState('yolo')
  const [yoloModel, setYoloModel]           = useState('')
  const [roboflowModelId, setRoboflowModelId] = useState('')
  const [classes, setClasses]               = useState('')
  const [classesHint, setClassesHint]       = useState('')
  const [conf, setConf]                     = useState('0.35')
  const [fps, setFps]                       = useState('1.0')
  const [device, setDevice]                 = useState('0')
  const [videoPath, setVideoPath]           = useState('')
  const [imagesPath, setImagesPath]         = useState('')
  const [existingPath, setExistingPath]     = useState('')
  const [skipTrain, setSkipTrain]           = useState(true)
  const [skipFilter, setSkipFilter]         = useState(false)
  const [outputOverride, setOutputOverride] = useState('')
  const [running, setRunning]               = useState(false)
  const [activeId, setActiveId]             = useState<string | null>(null)
  const [logs, setLogs]                     = useState<LogEntry[]>([])
  const [jobStatus, setJobStatus]           = useState('')
  const logOffsetRef = useRef(0)
  const logRef  = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const imageRef = useRef<HTMLInputElement>(null)

  // Auto-populate model classes on mount
  useEffect(() => {
    fetch('/api/model-classes').then(r => r.ok ? r.json() : null).then(d => {
      if (!d) return
      setClasses(d.classes.join(','))
      setClassesHint(`from ${d.model}`)
      setYoloModel(d.path ?? '')
    }).catch(() => {})
  }, [])

  // Load jobs list once on mount (single-user — list only changes on our actions)
  useEffect(() => {
    fetchJobs()
  }, [])

  // Poll active job logs while queued or running
  useEffect(() => {
    if (!activeId || !running) return
    const t = setInterval(() => pollJob(activeId), 2000)
    return () => clearInterval(t)
  }, [activeId, running])

  // Auto-scroll logs
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  async function fetchJobs() {
    const r = await fetch('/api/jobs')
    if (r.ok) setJobs(await r.json())
  }

  async function pollJob(id: string) {
    const r = await fetch(`/api/jobs/${id}?log_since=${logOffsetRef.current}`)
    if (!r.ok) return
    const d = await r.json()
    logOffsetRef.current = d.log_count
    setLogs(prev => [
      ...prev,
      ...(d.logs ?? []).map((l: string) => ({
        text: l,
        type: (l.startsWith('✓') || l.startsWith('▶') ? 'ok' : /❌|error/i.test(l) ? 'err' : 'default') as LogEntry['type'],
      })),
    ])
    setJobStatus(d.status)
    if (d.status === 'ready' || d.status === 'error') {
      setRunning(false)
      fetchJobs()
    } else if (d.status === 'queued' || d.status === 'running') {
      // keep polling — job is still in queue or actively running
      fetchJobs()
    }
  }

  async function handleBrowse(
    setter: (val: string) => void,
    currentValue: string,
    mode: 'folder' | 'file',
    fileTypes?: string
  ) {
    const qs = new URLSearchParams({
      mode,
      start: currentValue,
      file_types: fileTypes || ''
    })
    try {
      const r = await fetch(`/api/browse?${qs.toString()}`)
      if (r.ok) {
        const { path } = await r.json()
        if (path) setter(path)
      }
    } catch (err) {
      console.error('Failed to browse', err)
    }
  }

  async function loadClassesFromModel() {
    if (!yoloModel.trim()) {
      alert('Please enter or browse a model path first.')
      return
    }
    try {
      const r = await fetch(`/api/model-classes?path=${encodeURIComponent(yoloModel.trim())}`)
      if (r.ok) {
        const d = await r.json()
        if (d.classes && d.classes.length > 0) {
          setClasses(d.classes.join(','))
          setClassesHint(`from ${d.model}`)
        } else {
          alert('No classes found in model metadata.')
        }
      } else {
        const errText = await r.text()
        alert(`Failed to load classes: ${errText}`)
      }
    } catch (err) {
      console.error(err)
      alert('Error connecting to backend to fetch classes.')
    }
  }

  async function deleteJob(id: string, e: React.MouseEvent) {
    e.stopPropagation()
    if (!confirm('Are you sure you want to delete this job? This will remove it from the database.')) return
    try {
      const r = await fetch(`/api/jobs/${id}`, { method: 'DELETE' })
      if (r.ok) {
        if (activeId === id) {
          setActiveId(null)
          setLogs([])
        }
        fetchJobs()
      } else {
        alert('Failed to delete job')
      }
    } catch (err) {
      console.error(err)
      alert('Error deleting job')
    }
  }

  async function submit() {
    const fd = new FormData()
    if (tab === 'existing') {
      if (!existingPath.trim()) { alert('Enter pipeline output path'); return }
      fd.append('existing_output', existingPath)
    } else {
      if (inputMode === 'upload') {
        const f = fileRef.current?.files?.[0]
        if (!f) { alert('Select a file'); return }
        fd.append('video', f)
      } else if (inputMode === 'image') {
        const f = imageRef.current?.files?.[0]
        if (!f) { alert('Select an image'); return }
        fd.append('image', f)
      } else if (inputMode === 'video_path') {
        if (!videoPath.trim()) { alert('Enter video path'); return }
        fd.append('video_path', videoPath)
      } else {
        if (!imagesPath.trim()) { alert('Enter images directory'); return }
        fd.append('images_path', imagesPath)
      }
      fd.append('classes', labeler === 'yolo' ? (classes || 'placeholder') : classes)
      fd.append('labeler', labeler)
      fd.append('yolo_model', yoloModel)
      fd.append('roboflow_model_id', roboflowModelId)
      fd.append('conf', conf)
      fd.append('fps', fps)
      fd.append('device', device)
      fd.append('skip_training', skipTrain ? 'true' : 'false')
      fd.append('skip_filter', (skipFilter || inputMode === 'images_path' || inputMode === 'image') ? 'true' : 'false')
      fd.append('output_dir_override', outputOverride)
    }

    setRunning(true); setLogs([]); logOffsetRef.current = 0; setJobStatus('running')
    const r = await fetch('/api/jobs', { method: 'POST', body: fd })
    if (!r.ok) { alert(await r.text()); setRunning(false); return }
    const { job_id } = await r.json()
    setActiveId(job_id)
    fetchJobs()
  }

  const statusCls = (s: string) =>
    STATUS_CLS[s as keyof typeof STATUS_CLS] ?? 'bg-slate-800/60 text-slate-400 border border-slate-700/60'

  // Summary Metrics
  const totalJobs = jobs.length
  const completedJobs = jobs.filter(j => j.status === 'ready').length
  const runningJobs = jobs.filter(j => j.status === 'running' || j.status === 'queued').length
  const errorJobs = jobs.filter(j => j.status === 'error').length

  return (
    <div className="p-8 max-w-[1440px] mx-auto space-y-8 animate-in fade-in duration-300">
      
      {/* ── Dashboard Header / Summary Counters ────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Total Pipelines Run', count: totalJobs, color: 'from-blue-600 to-indigo-600', icon: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10' },
          { label: 'Completed & Ready', count: completedJobs, color: 'from-emerald-600 to-teal-600', icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z' },
          { label: 'Active Runs', count: runningJobs, color: 'from-indigo-600 to-violet-600', icon: 'M4 4v5h.582m15.356 2A8.001 8.001 0 1121.21 7.89M9 11l3-3m0 0l3 3m-3-3v8', pulse: runningJobs > 0 },
          { label: 'Pipeline Failures', count: errorJobs, color: 'from-rose-600 to-orange-600', icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z' },
        ].map((card, i) => (
          <div key={i} className="relative bg-[#0d0d16]/90 border border-[#1e2035]/60 rounded-2xl p-5 flex items-center gap-4 overflow-hidden group shadow-lg shadow-black/20">
            <div className={`absolute top-0 left-0 w-1 h-full bg-gradient-to-b ${card.color}`} />
            <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${card.color} flex items-center justify-center text-white shadow-md shadow-indigo-500/10`}>
              <svg className={`w-5 h-5 ${card.pulse ? 'animate-spin' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d={card.icon} />
              </svg>
            </div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">{card.label}</div>
              <div className="text-2xl font-black text-slate-100 tracking-tight mt-0.5">{card.count}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[440px_1fr] gap-8 items-start">
        
        {/* ── Left Column: Job Form ─────────────────────────────────────────── */}
        <div className="bg-[#0d0d16]/90 border border-[#1e2035]/70 rounded-2xl shadow-xl shadow-black/30 overflow-hidden">
          <div className="bg-gradient-to-r from-[#141424] to-[#0d0d16] px-6 py-4 border-b border-[#1e2035]/70 flex items-center justify-between">
            <div className="text-xs font-extrabold uppercase tracking-widest text-slate-300">
              Pipeline Control
            </div>
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
          </div>

          <div className="p-6 space-y-5">
            {/* Mode Selector */}
            <div className="flex bg-[#07070d] p-1 rounded-xl border border-[#1e2035]/50">
              {(['new', 'existing'] as const).map(t => (
                <button key={t} onClick={() => setTab(t)}
                  className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all duration-200 ${
                    tab === t
                      ? 'bg-gradient-to-r from-indigo-600 to-indigo-500 text-white shadow-md shadow-indigo-500/10'
                      : 'text-slate-500 hover:text-slate-300'
                  }`}>
                  {t === 'new' ? 'New Auto-Pipeline' : 'Load Existing Run'}
                </button>
              ))}
            </div>

            {tab === 'existing' ? (
              <div className="space-y-2">
                <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">Pipeline output directory</label>
                <div className="flex gap-2">
                  <input value={existingPath} onChange={e => setExistingPath(e.target.value)}
                    className="flex-1 bg-[#05050a] border border-[#1e2035] rounded-xl px-4 py-2.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500 font-mono"
                    placeholder="D:\yolo_world_poc\pipeline_output\run1" />
                  <button
                    onClick={() => handleBrowse(setExistingPath, existingPath, 'folder')}
                    type="button"
                    className="bg-[#18182b] hover:bg-[#20203a] text-slate-300 border border-[#1e2035] px-3.5 rounded-xl text-xs font-medium transition-all duration-150 whitespace-nowrap">
                    Browse
                  </button>
                </div>
                <p className="text-[10px] text-slate-500 italic mt-1 leading-normal">
                  Must contain a valid <code className="text-slate-400 font-semibold font-mono">2_filtered_frames/</code> output folder.
                </p>
              </div>
            ) : (
              <>
                {/* Input Type Selector */}
                <div className="space-y-2">
                  <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">Input Media Source</label>
                  <div className="flex gap-1.5 p-1 bg-[#07070d] rounded-xl border border-[#1e2035]/40">
                    {(['video_path', 'upload', 'images_path', 'image'] as const).map(m => (
                      <button key={m} onClick={() => {
                        setInputMode(m)
                        // Auto-enable skip-filter for pre-curated image sets
                        if (m === 'images_path' || m === 'image') setSkipFilter(true)
                        else setSkipFilter(false)
                      }}
                        className={`flex-1 text-[10px] font-bold py-2 rounded-lg transition-all duration-150 ${
                          inputMode === m
                            ? 'bg-[#18182b] text-indigo-400 border border-[#2b2b4d]/40'
                            : 'text-slate-500 hover:text-slate-300'
                        }`}>
                        {m === 'video_path' ? 'Video Path' : m === 'upload' ? 'Upload Video' : m === 'images_path' ? '🖼 Images Folder' : '🖼 Single Image'}
                      </button>
                    ))}
                  </div>
                </div>

                {inputMode === 'video_path' && (
                  <div className="space-y-2">
                    <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">Local Video file path</label>
                    <div className="flex gap-2">
                      <input value={videoPath} onChange={e => setVideoPath(e.target.value)}
                        className="flex-1 bg-[#05050a] border border-[#1e2035] rounded-xl px-4 py-2.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500 font-mono"
                        placeholder="D:\yolo_world_poc\videos\kitchen_feed.mp4" />
                      <button
                        onClick={() => handleBrowse(setVideoPath, videoPath, 'file', 'mp4,avi,mkv,mov')}
                        type="button"
                        className="bg-[#18182b] hover:bg-[#20203a] text-slate-300 border border-[#1e2035] px-3.5 rounded-xl text-xs font-medium transition-all duration-150 whitespace-nowrap">
                        Browse
                      </button>
                    </div>
                  </div>
                )}
                {inputMode === 'upload' && (
                  <div className="space-y-2">
                    <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">Upload video file</label>
                    <input ref={fileRef} type="file" accept="video/*"
                      className="w-full bg-[#05050a] border border-[#1e2035] rounded-xl px-4 py-2 text-xs text-slate-400 cursor-pointer file:mr-4 file:py-1 file:px-2 file:rounded-md file:border-0 file:text-[10px] file:font-semibold file:bg-indigo-950 file:text-indigo-400 hover:file:bg-indigo-900 file:transition-all" />
                  </div>
                )}
                {inputMode === 'images_path' && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">Images directory</label>
                      <span className="text-[9px] bg-emerald-950/40 text-emerald-400 border border-emerald-900/30 px-2 py-0.5 rounded-full">✓ Filtering skipped automatically</span>
                    </div>
                    <div className="flex gap-2">
                      <input value={imagesPath} onChange={e => setImagesPath(e.target.value)}
                        className="flex-1 bg-[#05050a] border border-[#1e2035] rounded-xl px-4 py-2.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500 font-mono"
                        placeholder="D:\myproject\frames" />
                      <button
                        onClick={() => handleBrowse(setImagesPath, imagesPath, 'folder')}
                        type="button"
                        className="bg-[#18182b] hover:bg-[#20203a] text-slate-300 border border-[#1e2035] px-3.5 rounded-xl text-xs font-medium transition-all duration-150 whitespace-nowrap">
                        Browse
                      </button>
                    </div>
                    <p className="text-[10px] text-slate-600">
                      All images go straight to the labeler — no blur/duplicate filtering applied.
                    </p>
                  </div>
                )}
                {inputMode === 'image' && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">Upload single image</label>
                      <span className="text-[9px] bg-emerald-950/40 text-emerald-400 border border-emerald-900/30 px-2 py-0.5 rounded-full">✓ Filtering skipped automatically</span>
                    </div>
                    <input ref={imageRef} type="file" accept="image/*"
                      className="w-full bg-[#05050a] border border-[#1e2035] rounded-xl px-4 py-2 text-xs text-slate-400 cursor-pointer file:mr-4 file:py-1 file:px-2 file:rounded-md file:border-0 file:text-[10px] file:font-semibold file:bg-indigo-950 file:text-indigo-400 hover:file:bg-indigo-900 file:transition-all" />
                    <p className="text-[10px] text-slate-600">
                      Test a single image — it goes straight to the labeler for annotation.
                    </p>
                  </div>
                )}

                {/* Model Configuration */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">Labeller engine</label>
                    <select value={labeler} onChange={e => setLabeler(e.target.value)}
                      className="w-full bg-[#05050a] border border-[#1e2035] rounded-xl px-3 py-2.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500 cursor-pointer">
                      <option value="yolo">YOLO (Trained Model)</option>
                      <option value="yolo_world">YOLO-World (Open-Vocab)</option>
                      <option value="grounding_dino">Grounding DINO</option>
                      <option value="roboflow">☁️ Roboflow Cloud</option>
                    </select>
                  </div>

                  {labeler !== 'roboflow' ? (
                    <div className="space-y-2">
                      <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">Compute Device</label>
                      <select value={device} onChange={e => setDevice(e.target.value)}
                        className="w-full bg-[#05050a] border border-[#1e2035] rounded-xl px-3 py-2.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500 cursor-pointer">
                        <option value="0">GPU (CUDA:0)</option>
                        <option value="cpu">CPU</option>
                      </select>
                    </div>
                  ) : (
                    <div className="flex items-end pb-0.5">
                      <div className="w-full bg-violet-950/20 border border-violet-900/30 rounded-xl px-3 py-2 text-[10px] text-violet-300">
                        ☁️ No GPU needed — runs on Roboflow's cloud
                      </div>
                    </div>
                  )}
                </div>

                {(labeler === 'yolo' || labeler === 'yolo_world') && (
                  <div className="space-y-2">
                    <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">YOLO Model weight path (.pt)</label>
                    <div className="flex gap-2">
                      <input value={yoloModel} onChange={e => setYoloModel(e.target.value)}
                        className="flex-1 bg-[#05050a] border border-[#1e2035] rounded-xl px-4 py-2.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500 font-mono"
                        placeholder="D:\yolo_world_poc\models\best.pt" />
                      <button
                        onClick={() => handleBrowse(setYoloModel, yoloModel, 'file', 'pt')}
                        type="button"
                        className="bg-[#18182b] hover:bg-[#20203a] text-slate-300 border border-[#1e2035] px-3.5 rounded-xl text-xs font-medium transition-all duration-150 whitespace-nowrap">
                        Browse
                      </button>
                    </div>
                  </div>
                )}

                {labeler === 'roboflow' && (
                  <div className="space-y-2">
                    <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">
                      Roboflow Model ID
                      <span className="ml-2 text-violet-400 normal-case font-normal">project-slug/version  (e.g. slcm-hawg5/1)</span>
                    </label>
                    <input
                      value={roboflowModelId}
                      onChange={e => setRoboflowModelId(e.target.value)}
                      className="w-full bg-[#05050a] border border-violet-900/40 rounded-xl px-4 py-2.5 text-xs text-violet-200 focus:outline-none focus:border-violet-500 font-mono placeholder:text-violet-900"
                      placeholder="project-slug/version  (e.g. slcm-hawg5/1)"
                    />
                    <div className="text-[10px] text-slate-600 leading-relaxed space-y-0.5">
                      <p>Go to <span className="text-violet-400">Roboflow → your project → Versions → Deploy</span> and copy the <strong className="text-slate-400">Model ID</strong>.</p>
                      <p className="text-amber-700">⚠ Do NOT include the workspace prefix — use <code className="text-violet-400">project/version</code>, not <code className="text-red-500/70">workspace/project</code>.</p>
                      <p>API key must be saved in{' '}<a href="/settings" className="text-violet-400 hover:text-violet-300 underline">Settings</a>.</p>
                    </div>
                  </div>
                )}

                {/* Vocabulary Classes */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">
                      Classes to annotate
                      {classesHint && <span className="text-indigo-400 font-semibold italic ml-1.5 lowercase">({classesHint})</span>}
                    </label>
                    {(labeler === 'yolo' || labeler === 'yolo_world') && (
                      <button
                        type="button"
                        onClick={loadClassesFromModel}
                        className="text-[9px] text-indigo-400 hover:text-indigo-300 bg-indigo-950/40 px-2 py-0.5 border border-indigo-900/30 rounded font-bold transition-all">
                        Load model classes
                      </button>
                    )}
                  </div>
                  <input value={classes} onChange={e => setClasses(e.target.value)}
                    className="w-full bg-[#05050a] border border-[#1e2035] rounded-xl px-4 py-2.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500"
                    placeholder="e.g. burger, fries, drink, nuggets" />
                </div>

                {/* Checkbox settings */}
                <div className="flex flex-col gap-2 py-1">
                  <div className="flex items-center gap-2.5">
                    <input type="checkbox" id="skip-train" checked={skipTrain}
                      onChange={e => setSkipTrain(e.target.checked)}
                      className="w-4 h-4 rounded border-[#1e2035] bg-[#05050a] text-indigo-600 focus:ring-indigo-500 cursor-pointer transition-all" />
                    <label htmlFor="skip-train" className="text-xs text-slate-400 font-semibold cursor-pointer">
                      Skip YOLOv8 Training Step
                    </label>
                  </div>
                  {inputMode !== 'images_path' && inputMode !== 'image' && (
                    <div className="flex items-center gap-2.5">
                      <input type="checkbox" id="skip-filter" checked={skipFilter}
                        onChange={e => setSkipFilter(e.target.checked)}
                        className="w-4 h-4 rounded border-[#1e2035] bg-[#05050a] text-emerald-500 focus:ring-emerald-500 cursor-pointer transition-all" />
                      <label htmlFor="skip-filter" className="text-xs text-slate-400 font-semibold cursor-pointer">
                        Skip Frame Filtering
                        <span className="ml-1.5 text-slate-600 font-normal">(use all frames as-is, no blur/duplicate removal)</span>
                      </label>
                    </div>
                  )}
                </div>

                {/* Advanced parameters */}
                <details className="group border border-[#1e2035]/40 rounded-xl p-3 bg-[#07070d]/50 transition-all">
                  <summary className="text-[10px] text-slate-400 uppercase tracking-widest font-bold cursor-pointer select-none flex items-center justify-between list-none">
                    <span>Advanced Tuning Configuration</span>
                    <span className="text-slate-600 group-open:rotate-180 transition-transform duration-200">▼</span>
                  </summary>
                  <div className="space-y-4 mt-3 pt-3 border-t border-[#1e2035]/40">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-1">
                        <label className="block text-[9px] text-slate-500 uppercase tracking-wider font-bold">
                          Confidence: <span className="text-indigo-400 font-semibold">{conf}</span>
                        </label>
                        <input type="range" min="0.1" max="0.9" step="0.05" value={conf}
                          onChange={e => setConf(e.target.value)} className="w-full accent-indigo-500 h-1 bg-[#1e2035] rounded-lg appearance-none cursor-pointer" />
                      </div>
                      <div className="space-y-1">
                        <label className="block text-[9px] text-slate-500 uppercase tracking-wider font-bold">
                          FPS Target: <span className="text-indigo-400 font-semibold">{fps}</span>
                        </label>
                        <input type="range" min="0.5" max="5" step="0.5" value={fps}
                          onChange={e => setFps(e.target.value)} className="w-full accent-indigo-500 h-1 bg-[#1e2035] rounded-lg appearance-none cursor-pointer" />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <label className="block text-[10px] text-slate-400 uppercase tracking-widest font-bold">Output directory override</label>
                      <div className="flex gap-2">
                        <input value={outputOverride} onChange={e => setOutputOverride(e.target.value)}
                          className="flex-1 bg-[#05050a] border border-[#1e2035] rounded-xl px-4 py-2 text-xs text-slate-300 focus:outline-none focus:border-indigo-500 font-mono"
                          placeholder="Leave empty to use base output folder" />
                        <button
                          onClick={() => handleBrowse(setOutputOverride, outputOverride, 'folder')}
                          type="button"
                          className="bg-[#18182b] hover:bg-[#20203a] text-slate-300 border border-[#1e2035] px-3.5 rounded-xl text-xs font-medium transition-all duration-150 whitespace-nowrap">
                          Browse
                        </button>
                      </div>
                    </div>
                  </div>
                </details>
              </>
            )}

            {/* Run Button */}
            <button onClick={submit} disabled={running}
              className="w-full bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 active:scale-[0.99] disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs uppercase tracking-wider font-extrabold py-3.5 rounded-xl shadow-lg shadow-indigo-500/10 transition-all duration-150 flex items-center justify-center gap-2">
              {running ? (
                <>
                  <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Processing Pipeline...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Run Annotation Pipeline
                </>
              )}
            </button>

            {/* Terminal output box */}
            {activeId && (
              <div className="space-y-2 border-t border-[#1e2035]/60 pt-4">
                <div className="flex items-center justify-between">
                  <span className={`text-[9px] uppercase tracking-wider font-extrabold px-2.5 py-0.5 rounded-full ${statusCls(jobStatus)}`}>
                    {jobStatus}
                  </span>
                  {jobStatus === 'ready' && (
                    <button onClick={() => router.push(`/review/${activeId}`)}
                      className="text-[10px] bg-emerald-600 hover:bg-emerald-500 font-bold text-white px-2.5 py-1 rounded-lg transition-colors flex items-center gap-1.5 shadow-md shadow-emerald-500/10">
                      Open Review
                      <span>→</span>
                    </button>
                  )}
                </div>
                <div ref={logRef}
                  className="bg-[#05050b] border border-[#1e2035]/80 rounded-xl p-3.5 h-44 overflow-y-auto font-mono text-[10px] leading-relaxed text-indigo-300 shadow-inner">
                  {logs.map((l, i) => (
                    <div key={i} className={
                      l.type === 'ok' ? 'text-emerald-400' : l.type === 'err' ? 'text-rose-400' : 'text-slate-500'
                    }>{l.text}</div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Right Column: Jobs Dashboard List ────────────────────────────── */}
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-extrabold uppercase tracking-widest text-slate-400">Annotation Jobs</h2>
            <span className="text-[10px] text-slate-500 font-bold bg-[#13131f] px-2.5 py-1 rounded border border-[#1e2035] shadow-sm">
              Auto-Refreshes
            </span>
          </div>

          {jobs.length === 0 ? (
            <div className="bg-[#0d0d16]/90 border border-[#1e2035]/60 rounded-2xl p-8 text-center shadow-lg shadow-black/20">
              <svg className="w-12 h-12 text-slate-700 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
              <div className="text-slate-500 text-sm font-medium">No pipeline jobs found.</div>
              <div className="text-slate-600 text-xs mt-1">Configure and run a new pipeline on the left to start auto-annotating.</div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {jobs.map(j => {
                const totalReviewed = j.saved + j.skipped
                const hasReviews = j.saved > 0 || j.skipped > 0
                const percentSaved = hasReviews ? Math.round((j.saved / totalReviewed) * 100) : 0
                const percentSkipped = hasReviews ? Math.round((j.skipped / totalReviewed) * 100) : 0

                return (
                  <div key={j.id}
                    className="relative bg-[#0d0d16]/90 border border-[#1e2035]/60 rounded-2xl p-5 hover:border-indigo-500/50 hover:bg-[#11111f]/90 transition-all duration-300 shadow-md hover:shadow-lg shadow-black/10 hover:shadow-indigo-500/5 group flex flex-col justify-between">
                    
                    {/* Job Card Header */}
                    <div className="space-y-2.5">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <h3 className="text-sm font-black text-slate-100 truncate tracking-tight group-hover:text-indigo-400 transition-colors" title={j.name}>
                            {j.name}
                          </h3>
                          <p className="text-[10px] text-slate-600 font-mono mt-0.5">
                            ID: {j.id} · {j.created_at.split('T')[0]}
                          </p>
                        </div>
                        <span className={`text-[9px] uppercase tracking-wider font-extrabold px-2.5 py-0.5 rounded-full ${statusCls(j.status)}`}>
                          {j.status}
                        </span>
                      </div>

                      {/* Stat Metrics Bar */}
                      <div className="grid grid-cols-2 gap-3 pt-2">
                        <div className="bg-[#06060c] p-2 rounded-xl border border-[#1e2035]/30">
                          <div className="text-[9px] text-slate-500 font-bold uppercase tracking-wider">Saved Frames</div>
                          <div className="text-sm font-extrabold text-emerald-400 mt-0.5">{j.saved}</div>
                        </div>
                        <div className="bg-[#06060c] p-2 rounded-xl border border-[#1e2035]/30">
                          <div className="text-[9px] text-slate-500 font-bold uppercase tracking-wider">Skipped Frames</div>
                          <div className="text-sm font-extrabold text-amber-500 mt-0.5">{j.skipped}</div>
                        </div>
                      </div>

                      {/* Visual progress slider bar */}
                      <div className="space-y-1.5 pt-2">
                        <div className="flex items-center justify-between text-[10px] text-slate-500 font-bold">
                          <span>Annotation Distribution</span>
                          <span className="text-indigo-400">{percentSaved}% Saved</span>
                        </div>
                        <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden flex">
                          {hasReviews ? (
                            <>
                              <div style={{ width: `${percentSaved}%` }} className="bg-emerald-500 h-full" />
                              <div style={{ width: `${percentSkipped}%` }} className="bg-amber-500 h-full" />
                            </>
                          ) : (
                            <div className="w-full bg-[#1e2035]/40 h-full" />
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Card Actions Footer */}
                    <div className="flex items-center justify-between pt-5 mt-4 border-t border-[#1e2035]/40">
                      {/* Delete button (trash icon) */}
                      <button
                        onClick={(e) => deleteJob(j.id, e)}
                        className="text-slate-600 hover:text-rose-500 hover:bg-rose-950/20 p-2 rounded-xl border border-transparent hover:border-rose-900/30 transition-all duration-150 flex items-center justify-center"
                        title="Delete pipeline job">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.8" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>

                      {/* Review button */}
                      {j.status === 'queued' ? (
                        <span className="text-[10px] text-amber-500 font-semibold italic flex items-center gap-1.5">
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          Queued
                        </span>
                      ) : j.status === 'running' ? (
                        <span className="text-[10px] text-slate-600 font-semibold italic flex items-center gap-1.5">
                          <svg className="animate-spin h-3.5 w-3.5 text-indigo-500" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          Processing...
                        </span>
                      ) : (
                        <button
                          onClick={() => router.push(`/review/${j.id}`)}
                          className="bg-[#18182b] hover:bg-indigo-600 text-indigo-400 hover:text-white font-bold text-xs px-4 py-2 border border-[#2b2b4d]/60 hover:border-indigo-500 rounded-xl transition-all duration-200 flex items-center gap-1.5 shadow-md">
                          Open Review
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                          </svg>
                        </button>
                      )}
                    </div>

                  </div>
                )
              })}
            </div>
          )}
        </div>

      </div>
    </div>
  )
}

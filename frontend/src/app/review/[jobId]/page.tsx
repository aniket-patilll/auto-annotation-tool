'use client'
import { useCallback, useEffect, useState } from 'react'
import Link from 'next/link'
import { useParams } from 'next/navigation'

type ImageItem = {
  name: string
  status: 'pending' | 'saved' | 'skipped'
  counts?: Record<string, number>
  total?: number
}
type JobInfo   = { name: string; status: string; saved: number; skipped: number; total: number }

export default function ReviewPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const [job, setJob]               = useState<JobInfo | null>(null)
  const [images, setImages]         = useState<ImageItem[]>([])
  const [filter, setFilter]         = useState<'all' | 'pending' | 'saved' | 'skipped'>('all')
  const [classes, setClasses]       = useState<string[]>([])
  const [classesDir, setClassesDir] = useState('')
  const [selClass, setSelClass]     = useState('')
  const [newClass, setNewClass]     = useState('')
  const [showNewCls, setShowNewCls] = useState(false)
  const [modalIdx, setModalIdx]     = useState<number | null>(null)
  const [boxesOnly, setBoxesOnly]   = useState(false)

  const filtered = filter === 'all' ? images : images.filter(i => i.status === filter)

  // Annotated image (with labels) vs. boxes-only render (no class text)
  const imgSrc = (name: string) =>
    `/api/jobs/${jobId}/${boxesOnly ? 'boxes' : 'image'}/${encodeURIComponent(name)}`

  useEffect(() => { loadAll() }, [jobId])

  async function loadAll() {
    const [jr, ir, sr] = await Promise.all([
      fetch(`/api/jobs/${jobId}`),
      fetch(`/api/jobs/${jobId}/images`),
      fetch('/api/settings'),
    ])
    if (jr.ok) setJob(await jr.json())
    if (ir.ok) setImages((await ir.json()).images ?? [])

    let defaultDir = ''
    if (sr.ok) {
      const settings = await sr.json()
      defaultDir = settings.classes_dir || ''
      setClassesDir(defaultDir)
    }

    if (defaultDir) {
      await loadClasses(defaultDir)
    }
  }

  async function loadClasses(dir: string) {
    const r = await fetch(`/api/classes?classes_dir=${encodeURIComponent(dir)}`)
    if (r.ok) {
      setClasses((await r.json()).classes ?? [])
    }
  }

  async function handleClassesDirChange(dir: string) {
    setClassesDir(dir)
    await loadClasses(dir)
  }

  const doSave = useCallback(async () => {
    if (modalIdx === null || !filtered[modalIdx]) return
    if (!selClass) { alert('Select a class first'); return }
    const img = filtered[modalIdx]
    const fd = new FormData()
    fd.append('job_id', jobId)
    fd.append('filename', img.name)
    fd.append('class_name', selClass)
    fd.append('classes_dir', classesDir)
    await fetch('/api/save', { method: 'POST', body: fd })
    setImages(prev => prev.map(i => i.name === img.name ? { ...i, status: 'saved' } : i))
    const nextPending = filtered.findIndex((im, idx) => idx > modalIdx && im.status === 'pending')
    setModalIdx(nextPending !== -1 ? nextPending : Math.min(modalIdx, filtered.length - 2))
  }, [modalIdx, filtered, selClass, jobId, classesDir])

  const doSkip = useCallback(async () => {
    if (modalIdx === null || !filtered[modalIdx]) return
    const img = filtered[modalIdx]
    const fd = new FormData()
    fd.append('job_id', jobId); fd.append('filename', img.name)
    await fetch('/api/skip', { method: 'POST', body: fd })
    setImages(prev => prev.map(i => i.name === img.name ? { ...i, status: 'skipped' } : i))
    setModalIdx(prev => prev !== null ? Math.min(prev + 1, filtered.length - 1) : null)
  }, [modalIdx, filtered, jobId])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (modalIdx === null) return
      if ((e.target as HTMLElement).tagName === 'INPUT') return
      if (e.key === 'ArrowLeft'  || e.key === 'a') setModalIdx(p => p !== null ? Math.max(0, p - 1) : null)
      if (e.key === 'ArrowRight' || e.key === 'd') setModalIdx(p => p !== null ? Math.min(filtered.length - 1, p + 1) : null)
      if (e.key === 's' || e.key === 'Enter') doSave()
      if (e.key === 'x') doSkip()
      if (e.key === 'b') setBoxesOnly(v => !v)
      if (e.key === 'Escape') setModalIdx(null)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [modalIdx, filtered, doSave, doSkip])

  async function createClass() {
    if (!newClass.trim()) return
    const fd = new FormData()
    fd.append('name', newClass.trim())
    fd.append('classes_dir', classesDir)
    await fetch('/api/classes', { method: 'POST', body: fd })
    const name = newClass.trim()
    setClasses(prev => [...prev, name].sort())
    setSelClass(name)
    setNewClass(''); setShowNewCls(false)
  }

  const dotColor  = (s: string) => s === 'saved' ? 'bg-green-500' : s === 'skipped' ? 'bg-amber-500' : 'bg-slate-600'
  const cardCls   = (s: string) => s === 'saved' ? 'border-green-700' : s === 'skipped' ? 'border-amber-700/60 opacity-60' : 'border-[#1e2035]'
  const saved     = images.filter(i => i.status === 'saved').length
  const skipped   = images.filter(i => i.status === 'skipped').length
  const pending   = images.length - saved - skipped

  return (
    <div className="p-6 max-w-[1400px] mx-auto">
      {/* Header */}
      <div className="flex items-center gap-4 mb-5 flex-wrap">
        <Link href="/"
          className="text-sm bg-[#1e2035] hover:bg-[#2a2a45] text-slate-300 px-3 py-1.5 rounded-md transition-colors">
          ← Back
        </Link>
        <div>
          <div className="text-base font-semibold text-slate-100">{job?.name ?? jobId}</div>
          <div className="text-[11px] text-slate-600">{jobId}</div>
        </div>
        <div className="flex gap-5 text-xs ml-2">
          <span><span className="text-green-400 font-semibold">{saved}</span> saved</span>
          <span><span className="text-amber-400 font-semibold">{skipped}</span> skipped</span>
          <span><span className="text-slate-400 font-semibold">{pending}</span> pending</span>
          <span><span className="text-slate-600">{images.length}</span> total</span>
        </div>
        <button onClick={loadAll}
          className="ml-auto text-xs bg-[#1e2035] hover:bg-[#2a2a45] text-slate-400 px-3 py-1.5 rounded-md transition-colors">
          ↻ Refresh
        </button>
      </div>

      {/* Target Directory Picker */}
      <div className="bg-[#13131f] border border-[#1e2035] rounded-xl p-4 mb-5 flex items-center justify-between gap-4">
        <div className="flex-1 min-w-0">
          <label className="block text-[10px] text-slate-500 uppercase tracking-widest mb-1.5 font-medium">
            Annotations Target Save Directory
          </label>
          <div className="flex gap-2">
            <input
              value={classesDir}
              onChange={e => handleClassesDirChange(e.target.value)}
              className="flex-1 bg-[#080810] border border-[#1e2035] rounded-md px-3 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-indigo-500 font-mono"
              placeholder="e.g. D:\yolo_world_poc\annotatted_classes"
            />
            <button
              onClick={async () => {
                const qs = new URLSearchParams({ mode: 'folder', start: classesDir })
                const r = await fetch(`/api/browse?${qs.toString()}`)
                if (r.ok) {
                  const { path } = await r.json()
                  if (path) handleClassesDirChange(path)
                }
              }}
              type="button"
              className="bg-[#1e2035] hover:bg-[#2a2a45] text-slate-300 border border-[#1e2035] px-4 py-1.5 rounded-md text-xs font-medium transition-colors whitespace-nowrap">
              Browse…
            </button>
          </div>
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-2 mb-5">
        {(['all', 'pending', 'saved', 'skipped'] as const).map(f => (
          <button key={f} onClick={() => setFilter(f)}
            className={`px-3.5 py-1 rounded-full text-xs border transition-all ${
              filter === f
                ? 'border-indigo-500 bg-indigo-950/40 text-indigo-300'
                : 'border-[#1e2035] text-slate-500 hover:text-slate-300'
            }`}>
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* Image grid */}
      {filtered.length === 0 ? (
        <div className="text-slate-600 text-sm">
          {images.length === 0
            ? 'No images yet — pipeline may still be running.'
            : 'No images match this filter.'}
        </div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-3">
          {filtered.map((img, idx) => (
            <div key={img.name} onClick={() => setModalIdx(idx)}
              className={`relative bg-[#13131f] border rounded-lg overflow-hidden cursor-pointer transition-all hover:-translate-y-0.5 hover:border-indigo-700 ${cardCls(img.status)}`}>
              {(img.total ?? 0) > 0 && (
                <span className="absolute top-1.5 right-1.5 z-10 text-[10px] font-bold bg-black/70 text-emerald-300 border border-emerald-800/50 px-1.5 py-0.5 rounded-full backdrop-blur-sm">
                  {img.total} det
                </span>
              )}
              <img
                src={`/api/jobs/${jobId}/image/${encodeURIComponent(img.name)}`}
                loading="lazy" alt=""
                className="w-full aspect-video object-cover bg-[#06060d] block" />
              <div className="px-2 py-1.5 flex items-center justify-between">
                <span className="text-[10px] text-slate-500 truncate max-w-[130px]" title={img.name}>
                  {img.name}
                </span>
                <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dotColor(img.status)}`} />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Modal */}
      {modalIdx !== null && filtered[modalIdx] && (
        <div
          className="fixed inset-0 bg-black/90 flex items-center justify-center z-50 p-4"
          onClick={e => e.target === e.currentTarget && setModalIdx(null)}>
          <div className="bg-[#13131f] border border-[#1e2035] rounded-xl w-full max-w-4xl max-h-[92vh] flex flex-col overflow-hidden">

            <div className="flex-1 flex items-center justify-center bg-[#06060d] min-h-0">
              <img
                src={imgSrc(filtered[modalIdx].name)}
                alt="" className="max-w-full max-h-[65vh] object-contain" />
            </div>

            {/* Per-class detection counts */}
            <div className="px-4 pt-3 flex items-center gap-2 flex-wrap border-t border-[#1e2035]">
              <span className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">Detected</span>
              {Object.keys(filtered[modalIdx].counts ?? {}).length === 0 ? (
                <span className="text-[11px] text-slate-600">No detections</span>
              ) : (
                Object.entries(filtered[modalIdx].counts ?? {})
                  .sort((a, b) => b[1] - a[1])
                  .map(([cls, n]) => (
                    <span key={cls}
                      className="text-[11px] bg-emerald-950/40 text-emerald-300 border border-emerald-900/40 px-2 py-0.5 rounded-full">
                      {cls} <span className="font-bold text-emerald-200">×{n}</span>
                    </span>
                  ))
              )}
              <span className="ml-auto text-[11px] text-slate-500">
                Total: <span className="font-bold text-slate-300">{filtered[modalIdx].total ?? 0}</span>
              </span>
            </div>

            <div className="p-4 flex items-center gap-3 flex-wrap">
              <span className="text-[11px] text-slate-500 flex-1 truncate min-w-0">
                {filtered[modalIdx].name}
              </span>

              {!showNewCls ? (
                <>
                  <select value={selClass} onChange={e => setSelClass(e.target.value)}
                    className="bg-[#080810] border border-[#1e2035] rounded-md px-3 py-1.5 text-sm text-slate-300 focus:outline-none focus:border-indigo-500 min-w-[160px]">
                    <option value="">Select class…</option>
                    {classes.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                  <button onClick={() => setShowNewCls(true)}
                    className="text-xs bg-[#1e2035] hover:bg-[#2a2a45] text-slate-300 px-3 py-1.5 rounded-md">
                    + New
                  </button>
                </>
              ) : (
                <div className="flex gap-2">
                  <input autoFocus value={newClass} onChange={e => setNewClass(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && createClass()}
                    className="bg-[#080810] border border-[#1e2035] rounded-md px-3 py-1.5 text-sm text-slate-300 focus:outline-none focus:border-indigo-500 w-44"
                    placeholder="New class name" />
                  <button onClick={createClass}
                    className="text-xs bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1.5 rounded-md">
                    Create
                  </button>
                  <button onClick={() => setShowNewCls(false)}
                    className="text-xs bg-[#1e2035] hover:bg-[#2a2a45] text-slate-300 px-3 py-1.5 rounded-md">
                    ✕
                  </button>
                </div>
              )}

              <button onClick={doSkip}
                className="text-xs bg-red-900/60 hover:bg-red-800/60 text-red-300 px-4 py-1.5 rounded-md">
                Skip <span className="text-red-600 ml-1">[X]</span>
              </button>
              <button onClick={doSave}
                className="text-xs bg-green-800/60 hover:bg-green-700/60 text-green-300 px-4 py-1.5 rounded-md">
                Save <span className="text-green-600 ml-1">[S]</span>
              </button>

              <button
                onClick={() => setBoxesOnly(v => !v)}
                title="Toggle class-name labels on the boxes (B)"
                className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded-md border transition-colors ${
                  boxesOnly
                    ? 'bg-indigo-950/50 border-indigo-600 text-indigo-300'
                    : 'bg-[#1e2035] border-[#1e2035] text-slate-400 hover:bg-[#2a2a45]'
                }`}>
                <span className={`w-7 h-3.5 rounded-full relative transition-colors ${boxesOnly ? 'bg-indigo-500' : 'bg-slate-600'}`}>
                  <span className={`absolute top-0.5 w-2.5 h-2.5 rounded-full bg-white transition-all ${boxesOnly ? 'left-4' : 'left-0.5'}`} />
                </span>
                Boxes only <span className="text-slate-600">[B]</span>
              </button>

              <div className="flex items-center gap-2 ml-auto">
                <button
                  onClick={() => setModalIdx(p => p !== null ? Math.max(0, p - 1) : null)}
                  className="w-8 h-8 flex items-center justify-center bg-[#1e2035] hover:bg-[#2a2a45] rounded-md text-slate-400">
                  ‹
                </button>
                <span className="text-[11px] text-slate-500 min-w-[60px] text-center">
                  {modalIdx + 1} / {filtered.length}
                </span>
                <button
                  onClick={() => setModalIdx(p => p !== null ? Math.min(filtered.length - 1, p + 1) : null)}
                  className="w-8 h-8 flex items-center justify-center bg-[#1e2035] hover:bg-[#2a2a45] rounded-md text-slate-400">
                  ›
                </button>
              </div>
            </div>

            <div className="px-4 pb-3 flex gap-5 text-[10px] text-slate-700">
              <span>← → / A D: navigate</span>
              <span>S / Enter: save</span>
              <span>X: skip</span>
              <span>B: boxes only</span>
              <span>Esc: close</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

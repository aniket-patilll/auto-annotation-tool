import type { Metadata } from 'next'
import './globals.css'
import Link from 'next/link'

export const metadata: Metadata = {
  title: 'YOLO-World & YOLO Annotation Dashboard',
  description: 'Industrial-grade computer vision auto-labelling and review pipeline.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#08080f] text-slate-300 min-h-screen font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
        <nav className="sticky top-0 z-40 bg-[#0d0d16]/80 backdrop-blur-md border-b border-[#1e2035]/60 px-8 h-14 flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2.5 group">
            <span className="w-6 h-6 rounded-lg bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-white text-[10px] font-bold shadow-md shadow-indigo-500/20 group-hover:scale-105 transition-transform duration-200">
              Y
            </span>
            <span className="text-slate-100 font-bold text-sm tracking-wide group-hover:text-indigo-400 transition-colors duration-200">
              YOLO annotation studio
            </span>
          </Link>
          
          <div className="h-4 w-[1px] bg-slate-800" />
          
          <span className="text-slate-500 text-xs hidden sm:inline-block font-medium">
            Automated Video & Image Annotation Tool
          </span>
          
          <div className="ml-auto flex items-center gap-6">
            <span className="text-[10px] uppercase tracking-wider text-slate-600 font-semibold bg-slate-900 px-2 py-0.5 rounded border border-slate-800/60">
              Local Mode
            </span>
            <Link href="/settings"
              className="text-xs font-medium text-slate-400 hover:text-indigo-400 transition-colors duration-200 flex items-center gap-1.5">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Settings
            </Link>
          </div>
        </nav>
        {children}
      </body>
    </html>
  )
}

import React, { useState, useEffect } from 'react';
import { PROJECT_PLAN } from './constants';
import { ProgressState } from './types';
import { FlattenedList } from './components/FlattenedList';
import { TerminalHeader } from './components/TerminalHeader';
import { ProgressBar } from './components/ProgressBar';

const STORAGE_KEY = 'cli_app_progress_v1';

const App: React.FC = () => {
  // State initialization
  const [progress, setProgress] = useState<ProgressState>(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    return saved ? JSON.parse(saved) : {};
  });

  const [totalProgress, setTotalProgress] = useState({ completed: 0, total: 0 });

  // Persistence
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
    
    // Calculate global totals
    const allSubtasks = PROJECT_PLAN.flatMap(w => w.days.flatMap(d => d.tasks.flatMap(t => t.subtasks)));
    const total = allSubtasks.length;
    const completed = allSubtasks.filter(t => progress[t.id]).length;
    
    setTotalProgress({ completed, total });
  }, [progress]);

  const toggleTask = (id: string) => {
    setProgress(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  return (
    <div className="min-h-screen bg-[#1d2021] text-[#ebdbb2] p-2 sm:p-6 flex flex-col items-center justify-center font-mono selection:bg-green-700 selection:text-white">
      
      {/* Main Terminal Window */}
      <div className="w-full max-w-4xl h-[90vh] border-2 border-gray-600 bg-[#0c0c0c] shadow-2xl rounded-sm flex flex-col relative z-10">
        
        {/* Window Controls (Fake) */}
        <div className="h-8 bg-gray-800 border-b border-gray-600 flex items-center px-4 justify-between">
          <div className="flex gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500 opacity-70"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500 opacity-70"></div>
            <div className="w-3 h-3 rounded-full bg-green-500 opacity-70"></div>
          </div>
          <div className="text-gray-400 text-xs">user@mac-air: ~/projects/protocol</div>
          <div className="w-8"></div>
        </div>

        {/* Content Area */}
        <div className="flex-1 p-4 overflow-hidden flex flex-col">
          <TerminalHeader />
          
          <div className="mb-6 px-2 py-3 border border-green-900 bg-green-900/10">
            <ProgressBar 
              completed={totalProgress.completed} 
              total={totalProgress.total} 
              width={40} 
              label="Overall Protocol Completion"
              className="text-sm sm:text-base"
            />
          </div>

          <FlattenedList progress={progress} toggleTask={toggleTask} />
        </div>

        {/* Status Line */}
        <div className="h-8 bg-[#3c3836] text-white flex items-center px-4 text-xs justify-between">
           <div className="flex gap-4">
             <span className="font-bold bg-[#ebdbb2] text-[#282828] px-2">NORMAL</span>
             <span>master*</span>
             <span>utf-8</span>
           </div>
           <div className="flex gap-4">
              <span>{Math.round((totalProgress.completed/totalProgress.total)*100)}%</span>
              <span>Ln {totalProgress.completed}, Col 1</span>
           </div>
        </div>
      </div>
      
      <div className="mt-4 text-xs text-gray-500 font-mono hidden sm:block">
        [↑/↓] Navigate • [Space] Toggle • [F11] Fullscreen
      </div>
    </div>
  );
};

export default App;
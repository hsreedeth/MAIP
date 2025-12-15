import React, { useCallback, useEffect, useRef } from 'react';
import { PROJECT_PLAN } from '../constants';
import { ProgressState } from '../types';
import { ProgressBar } from './ProgressBar';

interface FlattenedListProps {
  progress: ProgressState;
  toggleTask: (id: string) => void;
}

interface FlatItem {
  type: 'week' | 'day' | 'task' | 'subtask';
  id: string;
  text: string;
  data?: any;
  parentId?: string;
  index: number;
}

export const FlattenedList: React.FC<FlattenedListProps> = ({ progress, toggleTask }) => {
  const [selectedIndex, setSelectedIndex] = React.useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<(HTMLDivElement | null)[]>([]);

  // Flatten the structure for easier keyboard nav
  const flatList: FlatItem[] = [];
  let currentIndex = 0;

  PROJECT_PLAN.forEach((week) => {
    flatList.push({ type: 'week', id: week.id, text: week.title, data: week, index: currentIndex++ });
    
    week.days.forEach((day) => {
      flatList.push({ type: 'day', id: day.id, text: day.title, index: currentIndex++ });
      
      day.tasks.forEach((task) => {
        flatList.push({ type: 'task', id: task.id, text: task.title, index: currentIndex++ });
        
        task.subtasks.forEach((subtask) => {
          flatList.push({ type: 'subtask', id: subtask.id, text: subtask.text, parentId: task.id, index: currentIndex++ });
        });
      });
    });
  });

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => Math.min(prev + 1, flatList.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => Math.max(prev - 1, 0));
    } else if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      const item = flatList[selectedIndex];
      if (item.type === 'subtask') {
        toggleTask(item.id);
      }
    }
  }, [flatList, selectedIndex, toggleTask]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  useEffect(() => {
    // Scroll into view logic
    const currentEl = itemRefs.current[selectedIndex];
    if (currentEl && containerRef.current) {
      const container = containerRef.current;
      const itemTop = currentEl.offsetTop;
      const itemBottom = itemTop + currentEl.offsetHeight;
      const containerTop = container.scrollTop;
      const containerBottom = containerTop + container.offsetHeight;

      if (itemTop < containerTop) {
        container.scrollTop = itemTop;
      } else if (itemBottom > containerBottom) {
        container.scrollTop = itemBottom - container.offsetHeight;
      }
    }
  }, [selectedIndex]);

  return (
    <div className="flex-1 overflow-y-auto pr-2 relative" ref={containerRef}>
       {flatList.map((item, i) => {
         const isSelected = i === selectedIndex;
         
         if (item.type === 'week') {
            const week = item.data;
            const allSubtasks = week.days.flatMap((d: any) => d.tasks.flatMap((t: any) => t.subtasks));
            const completedCount = allSubtasks.filter((s: any) => progress[s.id]).length;
            const totalCount = allSubtasks.length;
            
            return (
               <div 
                key={item.id} 
                ref={el => itemRefs.current[i] = el}
                className={`mt-6 mb-2 pt-4 border-t border-dashed border-gray-700 ${isSelected ? 'bg-gray-800' : ''}`}
                onClick={() => setSelectedIndex(i)}
               >
                 <div className="flex justify-between items-center mb-1">
                   <h2 className="text-xl font-bold text-yellow-500 uppercase tracking-widest">
                     {isSelected && <span className="mr-2 text-green-500">{'>'}</span>}
                     {item.text}
                   </h2>
                   <span className="text-gray-500 text-sm hidden sm:inline">{week.subtitle}</span>
                 </div>
                 <ProgressBar completed={completedCount} total={totalCount} width={30} />
               </div>
            );
         }

         if (item.type === 'day') {
           return (
             <div 
                key={item.id} 
                ref={el => itemRefs.current[i] = el}
                className={`mt-4 mb-2 text-purple-400 font-bold ${isSelected ? 'bg-gray-800' : ''}`}
                onClick={() => setSelectedIndex(i)}
             >
                {isSelected ? '> ' : '  '}:: {item.text}
             </div>
           );
         }

         if (item.type === 'task') {
           return (
             <div 
                key={item.id} 
                ref={el => itemRefs.current[i] = el}
                className={`mt-2 mb-1 text-blue-400 pl-4 ${isSelected ? 'bg-gray-800' : ''}`}
                onClick={() => setSelectedIndex(i)}
             >
               {isSelected ? '> ' : '  '}-- {item.text}
             </div>
           );
         }

         if (item.type === 'subtask') {
           const isChecked = progress[item.id] || false;
           return (
             <div 
               key={item.id}
               ref={el => itemRefs.current[i] = el}
               onClick={() => {
                 setSelectedIndex(i);
                 toggleTask(item.id);
               }}
               className={`
                 cursor-pointer pl-8 py-1 transition-colors duration-75 flex items-start
                 ${isSelected ? 'bg-green-900 text-white' : 'text-gray-400 hover:text-gray-200'}
                 ${isChecked ? 'opacity-50' : ''}
               `}
             >
               <span className="mr-3 font-bold text-green-500">
                 [{isChecked ? 'x' : ' '}]
               </span>
               <span className={`${isChecked ? 'line-through decoration-gray-600' : ''}`}>
                 {item.text}
               </span>
               {isSelected && <span className="ml-auto text-xs animate-pulse opacity-70">PRESS SPACE</span>}
             </div>
           );
         }
         return null;
       })}
       <div className="h-20"></div> {/* Spacer */}
    </div>
  );
};
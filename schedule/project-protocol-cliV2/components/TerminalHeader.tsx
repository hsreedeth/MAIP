import React, { useEffect, useState } from 'react';

export const TerminalHeader: React.FC = () => {
  const [time, setTime] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTime(now.toLocaleTimeString('en-US', { hour12: false }));
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="border-b border-gray-700 pb-4 mb-6">
      <div className="flex justify-between items-end mb-2 text-xs text-gray-500 uppercase">
        <div>System: MAC-M1-AIR</div>
        <div>User: ANALYST</div>
        <div>{time}</div>
      </div>
      <pre className="text-green-500 font-bold leading-none text-xs sm:text-sm select-none">
{`
 ____  ____  _____  ____  _____  ___  _____  __     
(  _ \\(  _ \\(  _  )(_  _)(  _  )/ __)(  _  )(  )    
 )___/ )   / )(_)(   )(   )(_)(( (__  )(_)(  )(__   
(__)  (__\\_)(_____) (__) (_____)\\___)(_____)(____)  
                                      CLI v1.0.4
`}
      </pre>
      <div className="mt-2 text-gray-400 text-sm italic">
        &gt; ./launch_schedule.sh --mode=riced
      </div>
    </div>
  );
};
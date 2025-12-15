export interface SubTask {
  id: string;
  text: string;
}

export interface Task {
  id: string;
  title: string;
  subtasks: SubTask[];
}

export interface DayGroup {
  id: string;
  title: string;
  tasks: Task[];
}

export interface Week {
  id: string;
  title: string;
  subtitle: string;
  days: DayGroup[];
}

export interface ProgressState {
  [key: string]: boolean;
}
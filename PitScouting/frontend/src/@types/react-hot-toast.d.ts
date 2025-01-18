declare module 'react-hot-toast' {
  interface ToastOptions {
    duration?: number;
    position?: 'top-right' | 'top-center' | 'top-left' | 'bottom-right' | 'bottom-center' | 'bottom-left';
  }

  export const Toaster: React.FC<{ position?: ToastOptions['position'] }>;
  
  export const toast: {
    (message: string, options?: ToastOptions): void;
    success(message: string, options?: ToastOptions): void;
    error(message: string, options?: ToastOptions): void;
    dismiss(id?: string): void;
    remove(id?: string): void;
  };

  export default toast;
}
import React from 'react';
import { AlertTriangle } from 'lucide-react';

/**
 * ErrorBoundary Component
 * 
 * A React class component that catches JavaScript errors anywhere in its child 
 * component tree, logs those errors, and displays a fallback UI instead of crashing 
 * the entire application tree.
 * 
 * @component
 * @example
 * return (
 *   <ErrorBoundary>
 *     <MyComponent />
 *   </ErrorBoundary>
 * )
 */
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error('App crash:', error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-slate-900 px-4">
                    <div className="text-center max-w-md">
                        <div className="w-16 h-16 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center mx-auto mb-6">
                            <AlertTriangle className="w-8 h-8 text-red-600 dark:text-red-400" />
                        </div>
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Something went wrong</h1>
                        <p className="text-gray-500 dark:text-gray-400 mb-6">
                            An unexpected error occurred. Please refresh the page to try again.
                        </p>
                        <button
                            onClick={() => window.location.reload()}
                            className="bg-medical-600 hover:bg-medical-700 text-white px-6 py-2.5 rounded-lg font-medium transition"
                        >
                            Refresh Page
                        </button>
                    </div>
                </div>
            );
        }
        return this.props.children;
    }
}

export default ErrorBoundary;

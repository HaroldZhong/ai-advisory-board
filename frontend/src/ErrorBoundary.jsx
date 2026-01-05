import React from 'react';

export class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        this.setState({ error, errorInfo });
        console.error("Uncaught error:", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="p-8 max-w-2xl mx-auto mt-10 bg-red-50 border border-red-200 rounded-lg text-red-900">
                    <h1 className="text-2xl font-bold mb-4">Something went wrong</h1>
                    <p className="mb-4">The application crashed. Here is the error:</p>
                    <pre className="bg-white p-4 rounded border border-red-100 overflow-auto text-sm font-mono mb-4">
                        {this.state.error && this.state.error.toString()}
                    </pre>
                    <details className="text-xs text-red-700">
                        <summary className="cursor-pointer mb-2">Stack Trace</summary>
                        <pre className="whitespace-pre-wrap">
                            {this.state.errorInfo && this.state.errorInfo.componentStack}
                        </pre>
                    </details>
                </div>
            );
        }

        return this.props.children;
    }
}

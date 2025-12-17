import { useState } from 'react';
import { cn } from "@/lib/utils";
import {
    FileText, FileSpreadsheet, FileImage, File, FileCode,
    CheckCircle, AlertCircle, XCircle, Loader2, Sparkles
} from "lucide-react";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import {
    Popover,
    PopoverContent,
    PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { api } from '@/api';
import { useSettings } from '@/contexts/SettingsContext';

/**
 * Get icon component for file type
 */
function getFileIcon(mimeType, filename) {
    if (mimeType?.startsWith('image/')) return FileImage;
    if (mimeType?.includes('pdf')) return FileText;
    if (mimeType?.includes('spreadsheet') || mimeType?.includes('csv') || filename?.endsWith('.xlsx') || filename?.endsWith('.csv')) return FileSpreadsheet;
    if (mimeType?.includes('presentation') || filename?.endsWith('.pptx')) return FileText;
    if (mimeType?.includes('html') || mimeType?.includes('json')) return FileCode;
    return File;
}

/**
 * Get status indicator
 */
function StatusIndicator({ status }) {
    switch (status) {
        case 'processing':
            return <Loader2 className="h-3 w-3 animate-spin text-blue-500" />;
        case 'success':
            return <CheckCircle className="h-3 w-3 text-green-500" />;
        case 'partial':
            return <AlertCircle className="h-3 w-3 text-amber-500" />;
        case 'failed':
            return <XCircle className="h-3 w-3 text-red-500" />;
        default:
            return null;
    }
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (!bytes) return '';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * AttachmentPill - Compact display of an attached file
 */
export default function AttachmentPill({
    attachment,
    onClick,
    onRemove,
    onUpdate,
    showRemove = false,
    showEnhance = false,
    className
}) {
    const [isEnhancing, setIsEnhancing] = useState(false);
    const [popoverOpen, setPopoverOpen] = useState(false);
    const { settings } = useSettings();

    const { filename, mime_type, status, warning, stats, size_bytes, attachment_id } = attachment;
    const FileIcon = getFileIcon(mime_type, filename);

    // Check if PDF and needs enhancement
    const isPdf = mime_type?.includes('pdf') || filename?.toLowerCase().endsWith('.pdf');
    const canEnhance = showEnhance && isPdf && (status === 'partial' || status === 'failed');

    // Truncate long filenames
    const displayName = filename.length > 20
        ? filename.slice(0, 17) + '...' + filename.slice(-6)
        : filename;

    // Build tooltip content
    const tooltipLines = [filename];
    if (size_bytes) tooltipLines.push(formatFileSize(size_bytes));
    if (stats?.page_count) tooltipLines.push(`${stats.page_count} pages`);
    if (stats?.slide_count) tooltipLines.push(`${stats.slide_count} slides`);
    if (stats?.sheet_count) tooltipLines.push(`${stats.sheet_count} sheets`);
    if (warning) tooltipLines.push(`⚠️ ${warning}`);

    const handleEnhance = async (engine) => {
        if (!attachment_id) return;

        setIsEnhancing(true);
        try {
            // Use ZDR setting from user preferences
            const result = await api.enhanceAttachment(attachment_id, engine, settings.zdrEnabled);
            // Update parent with new status
            if (onUpdate) {
                onUpdate({
                    ...attachment,
                    status: result.status,
                    method: result.method,
                    stats: { ...stats, char_count: result.char_count },
                    warning: result.error,
                });
            }
            setPopoverOpen(false);
        } catch (error) {
            console.error('Enhancement failed:', error);
            alert(`Enhancement failed: ${error.message}`);
        } finally {
            setIsEnhancing(false);
        }
    };

    const pillContent = (
        <div
            className={cn(
                "inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs",
                "bg-muted/80 border border-border/50",
                "transition-colors hover:bg-muted",
                onClick && "cursor-pointer",
                status === 'failed' && "border-red-500/30 bg-red-500/10",
                status === 'partial' && "border-amber-500/30 bg-amber-500/10",
                className
            )}
            onClick={canEnhance ? undefined : onClick}
        >
            <FileIcon className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
            <span className="truncate max-w-[120px]">{displayName}</span>
            {isEnhancing ? (
                <Loader2 className="h-3 w-3 animate-spin text-blue-500" />
            ) : (
                <StatusIndicator status={status} />
            )}

            {showRemove && onRemove && (
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        onRemove();
                    }}
                    className="ml-0.5 hover:text-destructive"
                >
                    <XCircle className="h-3 w-3" />
                </button>
            )}
        </div>
    );

    // If can enhance, wrap in popover
    if (canEnhance) {
        return (
            <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
                <PopoverTrigger asChild>
                    {pillContent}
                </PopoverTrigger>
                <PopoverContent className="w-64 p-3" side="top">
                    <div className="space-y-2">
                        <div className="text-sm font-medium">Enhanced Extraction</div>
                        <p className="text-xs text-muted-foreground">
                            {warning || "Local extraction had issues. Try enhanced processing:"}
                        </p>
                        <div className="flex flex-col gap-2">
                            <Button
                                size="sm"
                                variant="outline"
                                className="justify-start gap-2"
                                disabled={isEnhancing}
                                onClick={() => handleEnhance('pdf-text')}
                            >
                                <Sparkles className="h-3 w-3" />
                                <span>Standard</span>
                                <span className="ml-auto text-xs text-green-600">Free</span>
                            </Button>
                            <Button
                                size="sm"
                                variant="outline"
                                className="justify-start gap-2"
                                disabled={isEnhancing}
                                onClick={() => handleEnhance('mistral-ocr')}
                            >
                                <Sparkles className="h-3 w-3" />
                                <span>OCR (for scans)</span>
                                <span className="ml-auto text-xs text-amber-600">$0.002/page</span>
                            </Button>
                        </div>
                    </div>
                </PopoverContent>
            </Popover>
        );
    }

    // Otherwise, wrap in tooltip
    return (
        <TooltipProvider>
            <Tooltip>
                <TooltipTrigger asChild>
                    {pillContent}
                </TooltipTrigger>
                <TooltipContent side="top" className="text-xs max-w-[200px]">
                    {tooltipLines.map((line, i) => (
                        <div key={i}>{line}</div>
                    ))}
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    );
}

/**
 * AttachmentPillList - Display multiple attachments
 */
export function AttachmentPillList({ attachments, onRemove, onUpdate, showRemove = false, showEnhance = false }) {
    if (!attachments || attachments.length === 0) return null;

    return (
        <div className="flex flex-wrap gap-1.5">
            {attachments.map((att, index) => (
                <AttachmentPill
                    key={att.attachment_id || index}
                    attachment={att}
                    showRemove={showRemove}
                    showEnhance={showEnhance}
                    onRemove={onRemove ? () => onRemove(index) : undefined}
                    onUpdate={onUpdate ? (updated) => onUpdate(index, updated) : undefined}
                />
            ))}
        </div>
    );
}

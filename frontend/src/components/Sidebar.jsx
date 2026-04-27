import { useState, useRef, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  PlusCircle,
  BarChart2,
  MessageSquare,
  MoreHorizontal,
  Pencil,
  Trash2,
  FolderPlus,
  FolderOpen,
  FolderClosed,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ArrowRightLeft,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { groupConversationsByFolder } from "@/utils/conversationOrganization";
import {
  RESPONSIVE_WIDTHS,
  getSidebarMode,
  readSidebarCollapsedPreference,
  writeSidebarCollapsedPreference,
} from "@/utils/responsiveLayout";

function SidebarTooltip({ label, enabled, children }) {
  if (!enabled) return children;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {children}
      </TooltipTrigger>
      <TooltipContent side="right" className="max-w-[240px]">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

function InlineRenameInput({ defaultValue, onConfirm, onCancel }) {
  const [value, setValue] = useState(defaultValue);
  const ref = useRef(null);

  useEffect(() => {
    ref.current?.focus();
    ref.current?.select();
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === "Enter") onConfirm(value.trim());
    if (e.key === "Escape") onCancel();
  };

  return (
    <Input
      ref={ref}
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onKeyDown={handleKeyDown}
      onBlur={() => onConfirm(value.trim())}
      className="h-7 text-sm px-2"
      onClick={(e) => e.stopPropagation()}
    />
  );
}

function RenameDialog({ open, title, defaultValue, onConfirm, onOpenChange }) {
  const [value, setValue] = useState(defaultValue);

  useEffect(() => {
    if (open) setValue(defaultValue);
  }, [defaultValue, open]);

  const handleConfirm = () => {
    const nextValue = value.trim();
    if (nextValue) {
      onConfirm(nextValue);
    } else {
      onOpenChange(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[360px]">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        <Input
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") handleConfirm();
            if (event.key === "Escape") onOpenChange(false);
          }}
          autoFocus
        />
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleConfirm} disabled={!value.trim()}>
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ConversationActionsMenu({
  title,
  conv,
  folders,
  onRenameStart,
  onDelete,
  onMove,
  compact = false,
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className={cn(
            "rounded hover:bg-muted transition-colors",
            compact
              ? "absolute right-0.5 top-1/2 -translate-y-1/2 bg-background/90 p-1 text-muted-foreground shadow-sm"
              : "opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 transition-opacity"
          )}
          onClick={(event) => event.stopPropagation()}
          onPointerDown={(event) => event.stopPropagation()}
          aria-label={`Conversation actions for ${title}`}
        >
          <MoreHorizontal className="h-4 w-4" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuItem onClick={onRenameStart}>
          <Pencil className="mr-2 h-4 w-4" /> Rename
        </DropdownMenuItem>
        {folders && folders.length > 0 && (
          <DropdownMenuSub>
            <DropdownMenuSubTrigger>
              <ArrowRightLeft className="mr-2 h-4 w-4" /> Move to...
            </DropdownMenuSubTrigger>
            <DropdownMenuSubContent>
              <DropdownMenuItem onClick={() => onMove(conv.id, null)}>
                No Folder
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              {folders.map((folder) => (
                <DropdownMenuItem key={folder.id} onClick={() => onMove(conv.id, folder.id)}>
                  <FolderClosed className="mr-2 h-3 w-3" /> {folder.name}
                </DropdownMenuItem>
              ))}
            </DropdownMenuSubContent>
          </DropdownMenuSub>
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem
          className="text-destructive focus:text-destructive"
          onClick={() => onDelete(conv)}
        >
          <Trash2 className="mr-2 h-4 w-4" /> Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function FolderActionsMenu({ folder, onRenameStart, onDeleteFolder, compact = false }) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className={cn(
            "rounded hover:bg-muted transition-colors",
            compact
              ? "absolute right-0.5 top-1/2 -translate-y-1/2 bg-background/90 p-1 text-muted-foreground shadow-sm"
              : "opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 transition-opacity"
          )}
          onClick={(event) => event.stopPropagation()}
          onPointerDown={(event) => event.stopPropagation()}
          aria-label={`Folder actions for ${folder.name}`}
        >
          <MoreHorizontal className={compact ? "h-4 w-4" : "h-3.5 w-3.5"} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-40">
        <DropdownMenuItem onClick={onRenameStart}>
          <Pencil className="mr-2 h-4 w-4" /> Rename
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          className="text-destructive focus:text-destructive"
          onClick={() => onDeleteFolder(folder.id)}
        >
          <Trash2 className="mr-2 h-4 w-4" /> Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function ConversationItem({
  conv,
  isActive,
  onSelect,
  onRename,
  onDelete,
  onMove,
  folders,
  compact = false,
}) {
  const [isRenaming, setIsRenaming] = useState(false);
  const [showCompactRenameDialog, setShowCompactRenameDialog] = useState(false);
  const title = conv.title || "New Conversation";

  const handleRenameConfirm = (newTitle) => {
    setIsRenaming(false);
    setShowCompactRenameDialog(false);
    if (newTitle && newTitle !== conv.title) onRename(conv.id, newTitle);
  };

  const item = (
    <div
      className={cn(
        "group relative flex items-center gap-1 rounded-md px-2 py-2 cursor-pointer transition-colors",
        compact && "h-9 justify-center px-0",
        isActive
          ? "bg-secondary text-secondary-foreground"
          : "hover:bg-muted/50 text-muted-foreground hover:text-foreground"
      )}
      onClick={() => onSelect(conv.id)}
      aria-label={compact ? title : undefined}
    >
      <MessageSquare className="h-4 w-4 shrink-0 opacity-60" />
      <div className={cn("flex-1 min-w-0", compact && "hidden")}>
        {isRenaming ? (
          <InlineRenameInput
            defaultValue={title}
            onConfirm={handleRenameConfirm}
            onCancel={() => setIsRenaming(false)}
          />
        ) : (
          <>
            <p className="text-sm truncate">{title}</p>
            <p className="text-xs text-muted-foreground/60">{conv.message_count} messages</p>
          </>
        )}
      </div>
      {!isRenaming && (
        <ConversationActionsMenu
          title={title}
          conv={conv}
          folders={folders}
          onRenameStart={() => {
            if (compact) {
              setShowCompactRenameDialog(true);
            } else {
              setIsRenaming(true);
            }
          }}
          onDelete={onDelete}
          onMove={onMove}
          compact={compact}
        />
      )}
    </div>
  );

  return (
    <>
      <SidebarTooltip
        enabled={compact}
        label={`${title} · ${conv.message_count} ${conv.message_count === 1 ? "message" : "messages"}`}
      >
        {item}
      </SidebarTooltip>
      <RenameDialog
        open={showCompactRenameDialog}
        title="Rename conversation"
        defaultValue={title}
        onConfirm={handleRenameConfirm}
        onOpenChange={setShowCompactRenameDialog}
      />
    </>
  );
}

function FolderGroup({
  folder,
  conversations,
  currentConversationId,
  onSelectConversation,
  onRenameConversation,
  onDeleteConversation,
  onMoveConversation,
  onRenameFolder,
  onDeleteFolder,
  folders,
  compact = false,
}) {
  const [isOpen, setIsOpen] = useState(true);
  const [isRenaming, setIsRenaming] = useState(false);

  const handleFolderRename = (newName) => {
    setIsRenaming(false);
    if (newName && newName !== folder.name) onRenameFolder(folder.id, newName);
  };

  if (compact) {
    return (
      <div className="mb-2">
        <div className="group relative">
          <SidebarTooltip enabled label={`${folder.name} · ${conversations.length} conversations`}>
            <button
              type="button"
              className="flex h-9 w-full items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground"
              onClick={() => setIsOpen(!isOpen)}
              aria-label={`${folder.name}, ${conversations.length} conversations`}
            >
              {isOpen ? (
                <FolderOpen className="h-4 w-4 text-primary/70" />
              ) : (
                <FolderClosed className="h-4 w-4 text-primary/70" />
              )}
            </button>
          </SidebarTooltip>
          <FolderActionsMenu
            folder={folder}
            onRenameStart={() => setIsRenaming(true)}
            onDeleteFolder={onDeleteFolder}
            compact
          />
        </div>
        {isOpen && conversations.length > 0 && (
          <div className="mt-1 space-y-1">
            {conversations.map((conv) => (
              <ConversationItem
                key={conv.id}
                conv={conv}
                isActive={conv.id === currentConversationId}
                onSelect={onSelectConversation}
                onRename={onRenameConversation}
                onDelete={onDeleteConversation}
                onMove={onMoveConversation}
                folders={folders}
                compact
              />
            ))}
          </div>
        )}
        <RenameDialog
          open={isRenaming}
          title="Rename folder"
          defaultValue={folder.name}
          onConfirm={handleFolderRename}
          onOpenChange={setIsRenaming}
        />
      </div>
    );
  }

  return (
    <div className="mb-1">
      <div
        className="group flex items-center gap-1 px-2 py-1.5 rounded-md cursor-pointer hover:bg-muted/50 transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? (
          <ChevronDown className="h-3 w-3 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground" />
        )}
        {isOpen ? (
          <FolderOpen className="h-4 w-4 shrink-0 text-primary/70" />
        ) : (
          <FolderClosed className="h-4 w-4 shrink-0 text-primary/70" />
        )}
        <div className="flex-1 min-w-0">
          {isRenaming ? (
            <InlineRenameInput
              defaultValue={folder.name}
              onConfirm={handleFolderRename}
              onCancel={() => setIsRenaming(false)}
            />
          ) : (
            <span className="text-sm font-medium truncate">{folder.name}</span>
          )}
        </div>
        <span className="text-xs text-muted-foreground/50 mr-1">{conversations.length}</span>
        {!isRenaming && (
          <FolderActionsMenu
            folder={folder}
            onRenameStart={() => setIsRenaming(true)}
            onDeleteFolder={onDeleteFolder}
          />
        )}
      </div>
      {isOpen && (
        <div className="ml-4 border-l border-border/40 pl-2 space-y-0.5 mt-0.5">
          {conversations.length === 0 ? (
            <p className="text-xs text-muted-foreground/50 py-1 px-2 italic">Empty folder</p>
          ) : (
            conversations.map((conv) => (
              <ConversationItem
                key={conv.id}
                conv={conv}
                isActive={conv.id === currentConversationId}
                onSelect={onSelectConversation}
                onRename={onRenameConversation}
                onDelete={onDeleteConversation}
                onMove={onMoveConversation}
                folders={folders}
                compact={compact}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}

export function SidebarContent({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  onShowAnalytics,
  onItemClick,
  folders = [],
  onCreateFolder,
  onRenameFolder,
  onDeleteFolder,
  onRenameConversation,
  onDeleteConversation,
  onMoveConversation,
  sidebarMode = "expanded",
  onToggleSidebar,
  canToggleSidebar = true,
}) {
  const [showNewFolderDialog, setShowNewFolderDialog] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [conversationPendingDelete, setConversationPendingDelete] = useState(null);
  const isCompact = sidebarMode !== "expanded";

  const handleConversationSelect = (id) => {
    onSelectConversation(id);
    onItemClick?.();
  };

  const handleNewConversation = () => {
    onNewConversation();
    onItemClick?.();
  };

  const handleShowAnalytics = () => {
    onShowAnalytics();
    onItemClick?.();
  };

  const handleCreateFolder = () => {
    if (newFolderName.trim()) {
      onCreateFolder?.(newFolderName.trim());
      setNewFolderName("");
      setShowNewFolderDialog(false);
    }
  };

  const handleConfirmDeleteConversation = async () => {
    if (!conversationPendingDelete) return;
    await onDeleteConversation?.(conversationPendingDelete.id);
    setConversationPendingDelete(null);
    onItemClick?.();
  };

  const {
    folderConversationMap,
    unfolderedConversations,
  } = groupConversationsByFolder(conversations, folders);

  return (
    <TooltipProvider delayDuration={150}>
      <div className="flex flex-col h-full">
      <div className={cn("space-y-3", isCompact ? "p-2" : "p-4")}>
        <div className={cn("flex items-center px-1", isCompact ? "flex-col gap-1 justify-center" : "justify-between")}>
          <div className={cn("flex items-center gap-2", isCompact && "justify-center")}>
            <img src="/favicon.png" alt="Logo" className="w-7 h-7 rounded-lg" />
            {!isCompact && <h1 className="text-lg font-bold tracking-tight">AI Advisory Board</h1>}
          </div>
          {onToggleSidebar && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={onToggleSidebar}
              disabled={!canToggleSidebar}
              title={canToggleSidebar ? (isCompact ? "Expand sidebar" : "Collapse sidebar") : "Widen the window to expand the sidebar"}
              aria-label={canToggleSidebar ? (isCompact ? "Expand sidebar" : "Collapse sidebar") : "Sidebar is compact at this width"}
            >
              {isCompact ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
            </Button>
          )}
        </div>
        <div className={cn("flex gap-2", isCompact && "flex-col items-center")}>
          <SidebarTooltip enabled={isCompact} label="New chat">
            <Button
              onClick={handleNewConversation}
              className={cn(isCompact ? "h-10 w-10 px-0" : "flex-1 justify-start")}
              size="sm"
              aria-label="New chat"
            >
              <PlusCircle className={cn("h-4 w-4", !isCompact && "mr-2")} />
              {!isCompact && "New Chat"}
            </Button>
          </SidebarTooltip>
          <SidebarTooltip enabled={isCompact} label="New folder">
            <Button
              onClick={() => setShowNewFolderDialog(true)}
              variant="outline"
              size="sm"
              className={cn(isCompact ? "h-10 w-10 px-0" : "px-2")}
              title="New Folder"
              aria-label="New folder"
            >
              <FolderPlus className="h-4 w-4" />
            </Button>
          </SidebarTooltip>
        </div>
        <SidebarTooltip enabled={isCompact} label="Analytics">
          <Button
            onClick={handleShowAnalytics}
            variant="ghost"
            className={cn(isCompact ? "h-10 w-10 px-0 mx-auto" : "w-full justify-start")}
            size="sm"
            aria-label="Analytics"
          >
            <BarChart2 className={cn("h-4 w-4", !isCompact && "mr-2")} />
            {!isCompact && "Analytics"}
          </Button>
        </SidebarTooltip>
      </div>

      <Separator />

      <ScrollArea className={cn("flex-1 py-2", isCompact ? "px-2" : "px-2")}>
        <div className="space-y-1 p-1">
          {/* Folder groups */}
          {folders.map((folder) => (
            <FolderGroup
              key={folder.id}
              folder={folder}
              conversations={folderConversationMap[folder.id] || []}
              currentConversationId={currentConversationId}
              onSelectConversation={handleConversationSelect}
              onRenameConversation={onRenameConversation}
              onDeleteConversation={setConversationPendingDelete}
              onMoveConversation={onMoveConversation}
              onRenameFolder={onRenameFolder}
              onDeleteFolder={onDeleteFolder}
              folders={folders}
              compact={isCompact}
            />
          ))}

          {/* Unfoldered conversations */}
          {unfolderedConversations.length > 0 && folders.length > 0 && !isCompact && (
            <div className="px-2 pt-3 pb-1">
              <p className="text-xs font-medium text-muted-foreground/60 uppercase tracking-wider">Conversations</p>
            </div>
          )}
          {unfolderedConversations.length === 0 && folders.length === 0 && (
            <div className="p-6 text-sm text-muted-foreground text-center">
              No conversations yet
            </div>
          )}
          {unfolderedConversations.map((conv) => (
            <ConversationItem
              key={conv.id}
              conv={conv}
              isActive={conv.id === currentConversationId}
              onSelect={handleConversationSelect}
              onRename={onRenameConversation}
              onDelete={setConversationPendingDelete}
              onMove={onMoveConversation}
              folders={folders}
              compact={isCompact}
            />
          ))}
        </div>
      </ScrollArea>

      {/* New Folder Dialog */}
      <Dialog open={showNewFolderDialog} onOpenChange={setShowNewFolderDialog}>
        <DialogContent className="sm:max-w-[360px]">
          <DialogHeader>
            <DialogTitle>New Folder</DialogTitle>
          </DialogHeader>
          <Input
            placeholder="Folder name..."
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreateFolder()}
            autoFocus
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowNewFolderDialog(false)}>Cancel</Button>
            <Button onClick={handleCreateFolder} disabled={!newFolderName.trim()}>Create</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(conversationPendingDelete)}
        onOpenChange={(open) => !open && setConversationPendingDelete(null)}
      >
        <DialogContent className="sm:max-w-[380px]">
          <DialogHeader>
            <DialogTitle>Delete conversation?</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            This permanently deletes "{conversationPendingDelete?.title || 'New Conversation'}".
            Attachments that are not used elsewhere will also be cleaned up.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConversationPendingDelete(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleConfirmDeleteConversation}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      </div>
    </TooltipProvider>
  );
}

/**
 * Desktop sidebar wrapper - hidden on mobile
 */
export default function Sidebar(props) {
  const [viewportWidth, setViewportWidth] = useState(() => (
    typeof window === "undefined" ? RESPONSIVE_WIDTHS.fullDesktop : window.innerWidth
  ));
  const [userCollapsed, setUserCollapsed] = useState(() => {
    if (typeof window === "undefined") return false;
    return readSidebarCollapsedPreference();
  });

  useEffect(() => {
    const handleResize = () => setViewportWidth(window.innerWidth);
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const sidebarMode = getSidebarMode(viewportWidth, userCollapsed);
  const isCompact = sidebarMode !== "expanded";
  const canToggleSidebar = viewportWidth >= RESPONSIVE_WIDTHS.fullDesktop;

  const handleToggleSidebar = () => {
    if (!canToggleSidebar) return;
    setUserCollapsed((current) => {
      const next = !current;
      writeSidebarCollapsedPreference(next);
      return next;
    });
  };

  return (
    <aside
      className={cn(
        "border-r bg-muted/10 hidden md:flex flex-col h-full transition-[width] duration-200 ease-out",
        isCompact ? "w-[64px]" : "w-[300px]"
      )}
      data-sidebar-mode={sidebarMode}
    >
      <SidebarContent
        {...props}
        sidebarMode={sidebarMode}
        onToggleSidebar={handleToggleSidebar}
        canToggleSidebar={canToggleSidebar}
      />
    </aside>
  );
}

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
  ChevronRight,
  ArrowRightLeft,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { groupConversationsByFolder } from "@/utils/conversationOrganization";

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

function ConversationItem({
  conv,
  isActive,
  onSelect,
  onRename,
  onDelete,
  onMove,
  folders,
}) {
  const [isRenaming, setIsRenaming] = useState(false);

  const handleRenameConfirm = (newTitle) => {
    setIsRenaming(false);
    if (newTitle && newTitle !== conv.title) onRename(conv.id, newTitle);
  };

  return (
    <div
      className={cn(
        "group flex items-center gap-1 rounded-md px-2 py-2 cursor-pointer transition-colors",
        isActive
          ? "bg-secondary text-secondary-foreground"
          : "hover:bg-muted/50 text-muted-foreground hover:text-foreground"
      )}
      onClick={() => onSelect(conv.id)}
    >
      <MessageSquare className="h-4 w-4 shrink-0 opacity-60" />
      <div className="flex-1 min-w-0">
        {isRenaming ? (
          <InlineRenameInput
            defaultValue={conv.title || "New Conversation"}
            onConfirm={handleRenameConfirm}
            onCancel={() => setIsRenaming(false)}
          />
        ) : (
          <>
            <p className="text-sm truncate">{conv.title || "New Conversation"}</p>
            <p className="text-xs text-muted-foreground/60">{conv.message_count} messages</p>
          </>
        )}
      </div>
      {!isRenaming && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 rounded hover:bg-muted transition-opacity"
              onClick={(e) => e.stopPropagation()}
            >
              <MoreHorizontal className="h-4 w-4" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuItem onClick={() => setIsRenaming(true)}>
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
                  {folders.map((f) => (
                    <DropdownMenuItem key={f.id} onClick={() => onMove(conv.id, f.id)}>
                      <FolderClosed className="mr-2 h-3 w-3" /> {f.name}
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
      )}
    </div>
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
}) {
  const [isOpen, setIsOpen] = useState(true);
  const [isRenaming, setIsRenaming] = useState(false);

  const handleFolderRename = (newName) => {
    setIsRenaming(false);
    if (newName && newName !== folder.name) onRenameFolder(folder.id, newName);
  };

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
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                className="opacity-0 group-hover:opacity-100 focus:opacity-100 p-1 rounded hover:bg-muted transition-opacity"
                onClick={(e) => e.stopPropagation()}
              >
                <MoreHorizontal className="h-3.5 w-3.5" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-40">
              <DropdownMenuItem onClick={(e) => { e.stopPropagation(); setIsRenaming(true); }}>
                <Pencil className="mr-2 h-4 w-4" /> Rename
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive focus:text-destructive"
                onClick={(e) => { e.stopPropagation(); onDeleteFolder(folder.id); }}
              >
                <Trash2 className="mr-2 h-4 w-4" /> Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
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
}) {
  const [showNewFolderDialog, setShowNewFolderDialog] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [conversationPendingDelete, setConversationPendingDelete] = useState(null);

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
    <div className="flex flex-col h-full">
      <div className="p-4 space-y-3">
        <div className="flex items-center justify-between px-1">
          <div className="flex items-center gap-2">
            <img src="/favicon.png" alt="Logo" className="w-7 h-7 rounded-lg" />
            <h1 className="text-lg font-bold tracking-tight">AI Advisory Board</h1>
          </div>
        </div>
        <div className="flex gap-2">
          <Button onClick={handleNewConversation} className="flex-1 justify-start" size="sm">
            <PlusCircle className="mr-2 h-4 w-4" />
            New Chat
          </Button>
          <Button
            onClick={() => setShowNewFolderDialog(true)}
            variant="outline"
            size="sm"
            className="px-2"
            title="New Folder"
          >
            <FolderPlus className="h-4 w-4" />
          </Button>
        </div>
        <Button onClick={handleShowAnalytics} variant="ghost" className="w-full justify-start" size="sm">
          <BarChart2 className="mr-2 h-4 w-4" />
          Analytics
        </Button>
      </div>

      <Separator />

      <ScrollArea className="flex-1 px-2 py-2">
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
            />
          ))}

          {/* Unfoldered conversations */}
          {unfolderedConversations.length > 0 && folders.length > 0 && (
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
  );
}

/**
 * Desktop sidebar wrapper - hidden on mobile
 */
export default function Sidebar(props) {
  return (
    <aside className="w-[300px] border-r bg-muted/10 hidden md:flex flex-col h-full">
      <SidebarContent {...props} />
    </aside>
  );
}

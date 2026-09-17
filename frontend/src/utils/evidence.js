// Turn-local aliases are resolved only against the message's saved snapshot.
export function remarkEvidenceLinks(snapshot) {
  return () => (tree) => {
    function visit(node) {
      if (!node.children || ['link', 'image', 'code', 'inlineCode', 'math', 'inlineMath'].includes(node.type)) return;
      node.children = node.children.flatMap((child) => {
        if (child.type !== 'text') { visit(child); return [child]; }
        const nodes = [];
        let start = 0;
        for (const match of child.value.matchAll(/\[S[1-9][0-9]*\.[1-9][0-9]*\]/g)) {
          if (!snapshot?.citations?.[match[0]]) continue;
          nodes.push({ type: 'text', value: child.value.slice(start, match.index) });
          nodes.push({ type: 'link', url: `#aab-citation-${match[0].slice(1, -1)}`, children: [{ type: 'text', value: match[0] }] });
          start = match.index + match[0].length;
        }
        return start ? [...nodes, { type: 'text', value: child.value.slice(start) }] : [child];
      });
    }
    visit(tree);
  };
}

export function getTurnPresentation(message) {
  const meta = message.metadata || {};
  const state = message.turn_state || {
    generation: meta.generation_state || 'unknown',
    persistence: meta.persistence_state || 'unknown',
    memory: meta.memory_state || 'unknown',
  };
  const invalid = meta.citation_checks?.items?.filter((item) => item.status !== 'valid_locator') || [];
  const incomplete = Boolean(meta.evidence_snapshot?.sources?.some((source) => source.status !== 'success' || source.omitted?.length));
  const needsAttention = Boolean(message.unconfirmed || message.interruption || invalid.length || incomplete
    || Object.values(state).includes('failed')
    || (state.generation === 'complete' && !['pending', 'saved'].includes(state.persistence)));
  const settled = state.generation === 'complete' && state.persistence === 'saved'
    && ['indexed', 'skipped'].includes(state.memory);
  const label = needsAttention ? 'Needs attention'
    : settled ? 'Saved'
    : state.persistence === 'saved' ? `Saved · memory ${state.memory}`
    : state.generation === 'complete' && state.persistence === 'pending' ? 'Saving…'
    : state.generation === 'pending' ? 'In progress' : 'Status unavailable';
  return { state, invalid, incomplete, needsAttention, settled, label };
}

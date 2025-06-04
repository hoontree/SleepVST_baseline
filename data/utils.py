import torch


def collate_fn(batch, pad_value=0, stack_labels=True, return_dict=True):
    """Shared collate function for variable length sleep datasets."""
    x_hw = [item['x_hw'] for item in batch]
    x_bw = [item['x_bw'] for item in batch]
    labels = [item['label'] for item in batch]
    subject_ids = [item['subject_id'] for item in batch]
    start_idxs = [item['start_idx'] for item in batch]

    max_len_hw = max(x.shape[0] for x in x_hw)
    max_len_bw = max(x.shape[0] for x in x_bw)
    max_len_label = max(x.shape[0] for x in labels)

    padded_x_hw = []
    padded_x_bw = []
    padded_labels = []

    for hw, bw, label in zip(x_hw, x_bw, labels):
        pad_hw = max_len_hw - hw.shape[0]
        pad_bw = max_len_bw - bw.shape[0]
        pad_label = max_len_label - label.shape[0]

        if pad_hw > 0:
            hw_pad = torch.full((pad_hw, hw.shape[1]), pad_value, dtype=hw.dtype, device=hw.device)
            hw = torch.cat([hw, hw_pad], dim=0)
        if pad_bw > 0:
            bw_pad = torch.full((pad_bw, bw.shape[1]), pad_value, dtype=bw.dtype, device=bw.device)
            bw = torch.cat([bw, bw_pad], dim=0)
        if pad_label > 0 and stack_labels:
            label_pad = torch.full((pad_label,), -100, dtype=label.dtype, device=label.device)
            label = torch.cat([label, label_pad], dim=0)

        padded_x_hw.append(hw)
        padded_x_bw.append(bw)
        padded_labels.append(label)

    x_hw_batch = torch.stack(padded_x_hw, dim=0)
    x_bw_batch = torch.stack(padded_x_bw, dim=0)
    labels_batch = torch.stack(padded_labels, dim=0) if stack_labels else padded_labels

    lengths = torch.tensor([x.shape[0] for x in x_hw])

    if return_dict:
        return {
            'x_hw': x_hw_batch,
            'x_bw': x_bw_batch,
            'label': labels_batch,
            'lengths': lengths,
            'subject_ids': subject_ids,
            'start_idxs': start_idxs,
        }
    else:
        return x_hw_batch, x_bw_batch, labels_batch, lengths, subject_ids, start_idxs


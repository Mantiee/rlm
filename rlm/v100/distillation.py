"""Predecessor KL on supervised tokens; shared frozen base, no second 12B allocation."""

from contextlib import contextmanager


def masked_kl(student, teacher, labels, temperature: float = 1.0):
    import torch
    import torch.nn.functional as functional

    if student.shape != teacher.shape or student.shape[:2] != labels.shape:
        raise ValueError("Student, teacher and label shapes differ")
    mask = labels[:, 1:] != -100
    if not mask.any():
        raise ValueError("No supervised tokens for distillation")
    # The t-th causal logit predicts label t+1. Process vocabulary in token blocks
    # to avoid an additional full [sequence, vocabulary] FP32 distribution.
    selected_student = student[:, :-1][mask]
    selected_teacher = teacher[:, :-1][mask]
    total = student.new_zeros((), dtype=torch.float32)
    for offset in range(0, selected_student.shape[0], 16):
        current = functional.log_softmax(
            selected_student[offset : offset + 16].float() / temperature, dim=-1
        )
        previous = functional.log_softmax(
            selected_teacher[offset : offset + 16].float() / temperature, dim=-1
        )
        total = total + functional.kl_div(current, previous, reduction="sum", log_target=True)
    return total * temperature**2 / selected_student.shape[0]


@contextmanager
def teacher_mode(model, predecessor_adapter: bool):
    parameters = {name: value.requires_grad for name, value in model.named_parameters()}
    training = model.training
    active = model.active_adapter
    try:
        model.eval()
        if predecessor_adapter:
            model.set_adapter("teacher")
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            yield
        else:
            with model.disable_adapter():
                yield
    finally:
        if predecessor_adapter:
            model.set_adapter(active)
        for name, value in model.named_parameters():
            value.requires_grad_(parameters[name])
        model.train(training)


def preserving_trainer(
    base_trainer, strength: float, temperature: float, predecessor_adapter: bool
):
    if not strength:
        return base_trainer

    class PreservingTrainer(base_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch

            # Teacher first. Restore the candidate before its forward and checkpointed
            # backward; otherwise recomputation could accidentally use teacher weights.
            teacher_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            with teacher_mode(model, predecessor_adapter), torch.no_grad():
                teacher = model(**teacher_inputs).logits.detach()
            outputs = model(**inputs)
            preservation = masked_kl(outputs.logits, teacher, inputs["labels"], temperature)
            loss = outputs.loss + strength * preservation
            if model.training:
                self.log(
                    {
                        "supervised_loss": outputs.loss.detach().item(),
                        "preservation_kl": preservation.detach().item(),
                    }
                )
            return (loss, outputs) if return_outputs else loss

    return PreservingTrainer

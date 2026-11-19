"""CPU-only checks; no dataset download, optimizer update or training run."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F
import tiktoken
from chat_format import format_chat
from prepare_sft import encode_example, examples_from_conversation
from train_sft import batch, lr_at
from model import GPT


class SFTDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.encoder = tiktoken.get_encoding('gpt2')

    def test_prompt_padding_are_ignored_but_reply_and_eos_are_supervised(self):
        prompt = format_chat([dict(role='user', content='Hello!')])
        ids, mask = encode_example(prompt, 'Hello there!', self.encoder)
        x, y = batch(dict(ids=ids[None], mask=mask[None]), np.array([0]), 'cpu')
        prefix = len(self.encoder.encode_ordinary(prompt))
        self.assertTrue(torch.all(y[0, :prefix - 1] == -100))
        self.assertEqual(int(y.ne(-100).sum()), len(self.encoder.encode_ordinary(' Hello there!')) + 1)
        self.assertEqual(int(y[0][y[0] != -100][-1]), self.encoder.eot_token)
        self.assertEqual(x.shape, (1, 512))
        self.assertEqual(int(y[0, -1]), -100)

    def test_long_answer_is_skipped_instead_of_getting_an_artificial_eos(self):
        self.assertIsNone(encode_example('User: Hi\nAssistant:', 'word ' * 100, self.encoder, seq_len=32))

    def test_multi_turn_format_matches_inference(self):
        messages = [dict(role='user', content=' Hi '), dict(role='assistant', content=' Hello '),
                    dict(role='user', content='What is your name?'), dict(role='assistant', content='A small model.')]
        samples = list(examples_from_conversation(messages))
        self.assertEqual(samples[-1][0], format_chat(messages[:-1]))
        self.assertEqual(samples[-1][0], 'User: Hi\nAssistant: Hello\nUser: What is your name?\nAssistant:')
        self.assertEqual(list(examples_from_conversation([dict(role='system', content='Special instruction'), *messages])), [])

    def test_learning_rate_has_warmup_and_complete_decay(self):
        values = [lr_at(s, 2500, 5e-5) for s in range(2500)]
        self.assertLess(values[0], max(values))
        self.assertAlmostEqual(max(values), 5e-5)
        self.assertAlmostEqual(values[-1], 5e-6)

    def test_masked_loss_uses_assistant_token_count(self):
        # Compare against explicit CE over only the two supervised positions.
        model = GPT().eval()
        captures = []
        handle = model.ln.register_forward_hook(lambda _, inp, out: captures.append(out))
        x = torch.tensor([[1, 2, 3, 4, 5, 6]])
        y = torch.tensor([[-100, 3, -100, 5, -100, -100]])
        with torch.no_grad():
            measured = model(x, y)
            handle.remove()
            logits = F.linear(captures[-1], model.wte.weight)
            keep = y.ne(-100)
            expected = F.cross_entropy(logits[keep], y[keep])
            self.assertTrue(torch.allclose(measured, expected, atol=1e-6))
            self.assertEqual(float(model(x, torch.full_like(y, -100))), 0.)


if __name__ == '__main__':
    unittest.main()

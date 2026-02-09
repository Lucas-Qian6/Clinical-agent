def verify_training_setup(model, ref_model, tokenizer, dpo_trainer, dataset):
    """
    Run this BEFORE dpo_trainer.train() to verify everything is set up correctly.
    Returns True if all checks pass, False otherwise.
    """
    print("\n" + "="*60)
    print("🔍 PRE-TRAINING VERIFICATION")
    print("="*60)

    all_checks_passed = True

    # ========== CHECK 1: Injector exists ==========
    print("\n[1/7] Checking injector exists...")
    if not hasattr(model, 'injector'):
        print("   ❌ FAIL: model.injector not found!")
        return False
    print(f"   ✅ Injector found (layer {model.injector.layer_idx}, alpha {model.injector.alpha})")

    # ========== CHECK 2: Injector parameters are in model ==========
    print("\n[2/7] Checking injector parameters are registered...")
    model_param_ids = set(id(p) for p in model.parameters())
    injector_param_ids = set(id(p) for p in model.injector.parameters())

    injector_in_model = injector_param_ids.issubset(model_param_ids)
    if not injector_in_model:
        print(f"   ❌ FAIL: {len(injector_param_ids - model_param_ids)} injector params NOT in model.parameters()!")
        print("   This means they won't be optimized!")
        all_checks_passed = False
    else:
        print(f"   ✅ All {len(injector_param_ids)} injector params registered in model")

    # Print injector param details
    for name, param in model.injector.named_parameters():
        print(f"      - {name}: shape={param.shape}, requires_grad={param.requires_grad}, device={param.device}")

    # ========== CHECK 3: Create optimizer and check injector parameters ==========
    print("\n[3/7] Creating optimizer and checking injector parameters...")

    # Force optimizer creation
    if dpo_trainer.optimizer is None:
        print("   Creating optimizer (trainer hasn't created it yet)...")
        dpo_trainer.create_optimizer()

    optimizer = dpo_trainer.optimizer

    if optimizer is None:
        print("   ❌ FAIL: Could not create optimizer!")
        return False

    optimizer_param_ids = set()
    for group_idx, group in enumerate(optimizer.param_groups):
        for p in group['params']:
            optimizer_param_ids.add(id(p))

    injector_in_optimizer = injector_param_ids.issubset(optimizer_param_ids)
    missing_in_opt = injector_param_ids - optimizer_param_ids

    if not injector_in_optimizer:
        print(f"   ❌ FAIL: {len(missing_in_opt)} injector params NOT in optimizer!")
        print("   These parameters will NOT update during training!")

        # Show which params are missing
        for name, param in model.injector.named_parameters():
            if id(param) in missing_in_opt:
                print(f"      Missing: {name}")

        all_checks_passed = False
    else:
        print(f"   ✅ All {len(injector_param_ids)} injector params in optimizer")

    # Print optimizer stats
    total_params_in_opt = len(optimizer_param_ids)
    lora_params = total_params_in_opt - len(injector_param_ids)
    print(f"      Total params in optimizer: {total_params_in_opt}")
    print(f"      - LoRA params: ~{lora_params}")
    print(f"      - Injector params: {len(injector_param_ids)}")

    # ========== CHECK 4: Reference model has injector (symmetry check) ==========
    print("\n[4/7] Checking reference model symmetry...")

    # Get actual ref model (may be stored differently)
    actual_ref_model = ref_model
    if actual_ref_model is None and hasattr(dpo_trainer, 'ref_model'):
        actual_ref_model = dpo_trainer.ref_model

    if actual_ref_model is None:
        print("   ℹ️  ref_model is None (trainer will create internal copy)")
        print("   ✅ This is OK: Policy uses emotion, reference doesn't.")
        print("   We're training the model to give emotion-aware responses vs baseline.")
        # This is intentional, not a failure
    elif not hasattr(actual_ref_model, 'injector'):
        print("   ℹ️  ref_model exists but has no injector")
        print("   ✅ Asymmetric setup: Policy=emotion-aware, Reference=baseline")
    else:
        # If ref model DOES have injector, check it's frozen
        ref_injector_trainable = any(p.requires_grad for p in actual_ref_model.injector.parameters())
        if ref_injector_trainable:
            print("   ❌ FAIL: ref_model.injector has trainable params!")
            print("   Reference model must be frozen!")
            all_checks_passed = False
        else:
            print("   ✅ ref_model has injector (frozen)")
            print("   Symmetric setup: Both models emotion-aware, only policy trains.")

    # ========== CHECK 5: Gradient flow test ==========
    print("\n[5/7] Testing gradient flow with dummy forward/backward...")
    model.train()

    # Get a sample
    sample = dataset[0]

    # Fix: ensure sample is dict with proper structure
    if isinstance(sample, str):
        # Dataset might be raw strings, need to convert
        print("   ⚠️  Dataset returns strings, not dicts. Skipping gradient test.")
        print("   Check your dataset format - should have 'prompt', 'chosen', 'rejected' keys")
        all_checks_passed = False
    else:
        try:
            # Create a proper batch using the collator
            batch = dpo_trainer.data_collator([sample])

            # Move to device
            device = next(model.parameters()).device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Store initial weights
            initial_weights = {name: param.clone().detach() for name, param in model.injector.named_parameters()}

            # Set emotion if available
            if 'emotion' in batch and hasattr(model, 'injector'):
                model.injector.set_emotion(batch['emotion'])

            # Simple forward pass (don't use concatenated_inputs, just do basic forward)
            # Use chosen response for testing
            outputs = model(
                input_ids=batch['chosen_input_ids'],
                attention_mask=batch['chosen_attention_mask'],
                labels=batch['chosen_input_ids'],  # Simple language modeling loss
            )

            loss = outputs.loss
            print(f"   Forward pass successful, loss={loss.item():.4f}")

            # Backward pass
            loss.backward()

            # Check gradients on injector
            grad_found = False
            grad_norms = []
            for name, param in model.injector.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    print(f"      - {name}: grad_norm={grad_norm:.6f}")
                    grad_found = True
                else:
                    print(f"      - {name}: ❌ NO GRADIENT!")

            if not grad_found:
                print("   ❌ FAIL: No gradients on injector parameters!")
                print("   Injector will NOT learn during training!")
                all_checks_passed = False
            else:
                avg_grad = sum(grad_norms) / len(grad_norms)
                print(f"   ✅ Gradients flowing to injector (avg norm: {avg_grad:.6f})")

                if avg_grad < 1e-8:
                    print("   ⚠️  WARNING: Gradient norms are very small - may indicate issues")

            # ========== CHECK 6: Optimizer step test ==========
            print("\n[6/7] Testing optimizer step updates injector weights...")

            # Clear previous gradients
            optimizer.zero_grad()

            # Do another forward/backward
            if 'emotion' in batch and hasattr(model, 'injector'):
                model.injector.set_emotion(batch['emotion'])

            outputs = model(
                input_ids=batch['chosen_input_ids'],
                attention_mask=batch['chosen_attention_mask'],
                labels=batch['chosen_input_ids'],
            )
            loss = outputs.loss
            loss.backward()

            # Take optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Check if weights changed
            weights_changed = False
            for name, param in model.injector.named_parameters():
                initial = initial_weights[name]
                diff = (param - initial).abs().max().item()
                if diff > 1e-8:
                    print(f"      - {name}: max_change={diff:.8f} ✅")
                    weights_changed = True
                else:
                    print(f"      - {name}: max_change={diff:.8f} ❌ NOT UPDATED!")

            if not weights_changed:
                print("   ❌ FAIL: Injector weights did NOT change after optimizer.step()!")
                print("   Training will not work!")
                all_checks_passed = False
            else:
                print("   ✅ Optimizer step successfully updated injector weights")

        except Exception as e:
            print(f"   ❌ FAIL: Error during gradient/optimizer test: {e}")
            import traceback
            traceback.print_exc()
            all_checks_passed = False

    # ========== CHECK 7: Data collator emotion handling ==========
    print("\n[7/7] Testing data collator emotion handling...")

    # Get samples and check format
    test_samples = [dataset[i] for i in range(min(3, len(dataset)))]

    # Check sample format
    first_sample = test_samples[0]
    if isinstance(first_sample, str):
        print("   ❌ FAIL: Dataset returns strings, not dicts!")
        print("   Expected dict with keys: 'prompt', 'chosen', 'rejected', 'emotion'")
        print(f"   Got: {type(first_sample)}")
        all_checks_passed = False
    elif isinstance(first_sample, dict):
        print(f"   Sample keys: {list(first_sample.keys())}")

        try:
            batch = dpo_trainer.data_collator(test_samples)

            if 'emotion' not in batch:
                print("   ⚠️  WARNING: 'emotion' key not in batch!")
                print(f"   Batch keys: {list(batch.keys())}")
                # Don't fail on this - emotion might be optional
            else:
                emotion_shape = batch['emotion'].shape
                expected_shape = (len(test_samples), 4)
                if emotion_shape != expected_shape:
                    print(f"   ⚠️  WARNING: emotion shape {emotion_shape} != expected {expected_shape}")
                else:
                    print(f"   ✅ Emotion in batch with correct shape: {emotion_shape}")
                    print(f"      Sample emotions:\n{batch['emotion'][:2]}")

        except Exception as e:
            print(f"   ❌ FAIL: Error in data collator: {e}")
            import traceback
            traceback.print_exc()
            all_checks_passed = False
    else:
        print(f"   ❌ FAIL: Unexpected sample type: {type(first_sample)}")
        all_checks_passed = False

    # ========== FINAL VERDICT ==========
    print("\n" + "="*60)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - Ready to train!")
        print("="*60)
        return True
    else:
        print("❌ SOME CHECKS FAILED - DO NOT START TRAINING")
        print("   Fix the issues above before proceeding.")
        print("="*60)
        return False
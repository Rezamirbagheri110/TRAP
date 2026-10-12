"""
Template Loader - Loads enhanced templates into DynamicPromptGenerator
"""

import logging

logger = logging.getLogger(__name__)


def load_enhanced_templates_into_generator(generator):
    """
    Load enhanced templates into an existing DynamicPromptGenerator instance

    Args:
        generator: DynamicPromptGenerator instance to enhance

    Returns:
        int: Number of templates added
    """
    try:
        import os
        from prompts.enhanced_templates import get_enhanced_templates
        from prompts.optimized_templates import get_optimized_templates_priority_map

        enhanced = get_enhanced_templates()
        priority_map = get_optimized_templates_priority_map()
        added_count = 0

        # CRITICAL FIX: Boost priority templates from config and priority map
        priority_templates_str = os.getenv("OURS_PRIORITY_TEMPLATES", "")
        priority_templates = [t.strip() for t in priority_templates_str.split(",") if t.strip()]

        # Add priority templates from priority map
        for template_name, priority_score in priority_map.items():
            if template_name not in priority_templates:
                priority_templates.append(template_name)

        # Ban defensive templates from config
        banned_templates_str = os.getenv("OURS_BANNED_TEMPLATES", "")
        banned_templates = [t.strip() for t in banned_templates_str.split(",") if t.strip()]

        # Removed most defensive keyword filters to restore diversity
        # Old code banned too many templates - now only ban obvious refusal-triggering phrases
        defensive_keywords = [
            'this is unethical', 'i cannot help', 'i must refuse',
            'this violates policy', 'against my guidelines'
        ]

        banned_count = 0

        # CRITICAL: First, remove banned templates from existing generator
        templates_to_remove = []
        for name, template in generator.templates.items():
            # Check if template is banned by name
            if name in banned_templates:
                templates_to_remove.append(name)
                continue

            # Don't filter based on characteristics - allow all templates for diversity
            # Removed characteristic-based and keyword-based filtering

        # Remove banned templates
        for name in templates_to_remove:
            if name in generator.templates:
                del generator.templates[name]
                banned_count += 1
                logger.warning(f"[BANNED] Removed existing defensive template: {name}")


        # Merge with existing templates
        for name, template in enhanced.items():
            # Check if template is banned by name
            if name in banned_templates:
                logger.warning(f"[BANNED] Skipping banned template: {name}")
                banned_count += 1
                continue

            # Don't filter based on characteristics - allow all templates for diversity
            # Only check for explicit banned list, not defensive characteristics

            if name not in generator.templates:
                generator.templates[name] = template
                added_count += 1

                # Boost effectiveness based on priority map
                if name in priority_map:
                    priority_score = priority_map[name]
                    template.effectiveness_score = max(template.effectiveness_score, priority_score)
                    logger.info(f"[PRIORITY] Boosted template: {name} (priority_score={priority_score:.2f}, eff={template.effectiveness_score:.2f})")
                elif name in priority_templates:
                    template.effectiveness_score = max(template.effectiveness_score, 8.5)
                    logger.info(f"[PRIORITY] Boosted template: {name} (eff={template.effectiveness_score:.2f})")
                else:
                    logger.info(f"[ENHANCED] Added template: {name} (eff={template.effectiveness_score:.2f})")

        logger.info(f"[ENHANCED] Successfully loaded {added_count}/{len(enhanced)} enhanced templates")
        if banned_count > 0:
            logger.warning(f"[BANNED] Skipped {banned_count} defensive/banned templates")
        logger.info(f"[PRIORITY] {len(priority_templates)} templates marked as high-priority")
        return added_count

    except ImportError as e:
        logger.warning(f"[ENHANCED] Could not import enhanced_templates: {e}")
        return 0
    except Exception as e:
        logger.error(f"[ENHANCED] Error loading enhanced templates: {e}")
        return 0


def create_enhanced_prompt_system():
    """
    Create DynamicPromptGenerator with enhanced templates pre-loaded

    Returns:
        DynamicPromptGenerator instance with all templates
    """
    from prompts.advanced_prompt_templates import DynamicPromptGenerator

    generator = DynamicPromptGenerator()
    load_enhanced_templates_into_generator(generator)

    return generator

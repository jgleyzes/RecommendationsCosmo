from django.contrib import admin
from app_compare.models import Article,Author,Suggestions,Tags
# Register your models here.

@admin.register(Author)
class AuthorAdmin(admin.ModelAdmin):
    """Author admin."""

    list_display = ('name', 'get_article')
    search_fields = ('name',)

    def get_article(self, obj):
        return obj.article_set.all()


admin.site.register(Article)
admin.site.register(Suggestions)
admin.site.register(Tags)
